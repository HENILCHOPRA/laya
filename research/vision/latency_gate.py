"""Laya-V hour-2 latency gate. Run on the TARGET inference GPU (Colab T4 / L4), not the A100.

Measures p50/p95 end-to-end latency for one image + 1, 5 and 10 questions through the full
Laya-V path: SigLIP 2 NaFlex tower (576 patches) -> resampler (128 tokens) -> splice into each
question row -> laya-multilingual encoder + decision head. Weights of the resampler are random
(latency does not depend on them). Compares So400m/16 and B/16; the gate locks So400m if its
1-question p50 is <= 100 ms.

Colab:
    !pip -q install -U laya "transformers>=4.56" pillow
    !python latency_gate.py
"""
import json
import os
import statistics
import time

import torch
import torch.nn as nn

ENCODERS = {"so400m": "google/siglip2-so400m-patch16-naflex", "b16": "google/siglip2-base-patch16-naflex"}
PATCHES = 576
SLOTS = 130
REPEATS = int(os.environ.get("GATE_REPEATS", "50"))
# Dry run on CPU (correctness only, timings meaningless): GATE_DEVICE=cpu GATE_ENCODERS=b16
DEVICE = os.environ.get("GATE_DEVICE", "cuda")
if os.environ.get("GATE_ENCODERS"):
    ENCODERS = {k: v for k, v in ENCODERS.items() if k in os.environ["GATE_ENCODERS"].split(",")}


class Layer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_s, self.ln_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, h, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, kv, pad):
        kv = self.ln_kv(kv)
        x = self.ln_q(q)
        q = q + self.cross(x, kv, kv, key_padding_mask=pad, need_weights=False)[0]
        x = self.ln_s(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        return q + self.ff(self.ln_f(q))


class Adapter(nn.Module):
    def __init__(self, in_dim, d=768):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d))
        self.queries = nn.Parameter(torch.randn(128, d) * 0.02)
        self.layers = nn.ModuleList([Layer(d, 12) for _ in range(2)])
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.se = nn.Parameter(torch.zeros(2, d))

    def forward(self, feats, mask):
        kv = self.inp(feats)
        q = self.queries.unsqueeze(0).expand(feats.size(0), -1, -1).to(kv.dtype)
        for l in self.layers:
            q = l(q, kv, ~mask.bool())
        x = self.proj(self.norm(q))
        se = self.se.to(x.dtype)
        return torch.cat([se[:1].expand(x.size(0), 1, -1), x, se[1:].expand(x.size(0), 1, -1)], 1)


def naflex(img, max_patches, patch=16):
    """SigLIP 2 NaFlex preprocessing (same as Siglip2ImageProcessor), without torchvision."""
    import math

    import numpy as np
    from PIL import Image

    def scaled(scale, size):
        return max(patch, int(math.ceil(size * scale / patch) * patch))

    lo, hi = 1e-6, 100.0
    while hi - lo >= 1e-5:
        mid = (lo + hi) / 2
        if (scaled(mid, img.height) / patch) * (scaled(mid, img.width) / patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    th, tw = scaled(lo, img.height), scaled(lo, img.width)
    arr = (np.asarray(img.resize((tw, th), Image.BILINEAR), dtype=np.float32) / 255.0 - 0.5) / 0.5
    gh, gw = th // patch, tw // patch
    pv = arr.reshape(gh, patch, gw, patch, 3).transpose(0, 2, 1, 3, 4).reshape(gh * gw, -1)
    out = np.zeros((max_patches, pv.shape[1]), dtype=np.float32)
    out[: len(pv)] = pv
    mask = np.zeros((max_patches,), dtype=np.int64)
    mask[: len(pv)] = 1
    return {"pixel_values": torch.from_numpy(out)[None], "pixel_attention_mask": torch.from_numpy(mask)[None],
            "spatial_shapes": torch.tensor([[gh, gw]])}


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    import laya
    from laya.common import QTYPES, build_sequence, collate_items
    from PIL import Image
    from transformers import Siglip2VisionModel

    dev = torch.device(DEVICE)
    if dev.type == "cuda":
        assert torch.cuda.is_available(), "run on a GPU runtime"
        gpu = torch.cuda.get_device_name(0)
        fp = torch.float16 if torch.cuda.get_device_capability(0)[0] < 8 else torch.bfloat16
    else:
        gpu, fp = "cpu (dry run)", torch.float32
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    agent = laya.Agent("convaiinnovations/laya", subfolder="multilingual", device=DEVICE)
    model = agent.model.eval()
    img = Image.new("RGB", (1024, 768), (180, 120, 60))
    qs = [{"t": "noul", "ins": "Is the receipt total above 50?", "crit": None},
          {"t": "choice", "ins": "What was bought?", "crit": {"meals": "food and drink", "travel": "transport",
                                                              "other": "everything else"}},
          {"t": "score", "ins": "How urgent is this?", "crit": ["not urgent", "soon", "blocking"]}]
    state = {"note": "Reimbursement claim for a team lunch with a client."}
    results = {"gpu": gpu, "dtype": str(fp), "patches": PATCHES}
    for name, repo in ENCODERS.items():
        tower = Siglip2VisionModel.from_pretrained(repo, torch_dtype=fp).to(dev).eval()
        adapter = Adapter(tower.config.hidden_size).to(dev, fp).eval()
        emb_layer = model.encoder.get_input_embeddings()
        for nq in (1, 5, 10):
            items = []
            sep = agent.tok.sep_token_id
            for i in range(nq):
                # Same layout as the Laya-V build_sequence(image_slots=130): slots right after the
                # option block's [SEP] (the second [SEP]), state text after them.
                ids, markers = build_sequence(agent.tok, state, qs[i % 3], 1024 - SLOTS, 256)
                # first [SEP] after the last option marker (mmBERT uses one id for [CLS] and [SEP])
                pos = next(j for j in range(markers[-1], len(ids)) if ids[j] == sep) + 1
                ids = ids[:pos] + [agent.tok.pad_token_id] * SLOTS + ids[pos:]
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES[qs[i % 3]["t"]], "pos": pos})
            b = collate_items([items], agent.tok.pad_token_id)
            pos = torch.tensor([it["pos"] for it in items], device=dev)
            times = []
            for r in range(REPEATS + 5):
                sync()
                t0 = time.perf_counter()
                with torch.no_grad(), torch.autocast(dev.type, dtype=fp, enabled=dev.type == "cuda"):
                    px = naflex(img, PATCHES)
                    feats = tower(pixel_values=px["pixel_values"].to(dev, fp),
                                  pixel_attention_mask=px["pixel_attention_mask"].to(dev),
                                  spatial_shapes=px["spatial_shapes"].to(dev)).last_hidden_state
                    vis = adapter(feats, px["pixel_attention_mask"].to(dev))
                    ids = b["input_ids"].to(dev)
                    emb = emb_layer(ids)
                    idx = pos[:, None] + torch.arange(SLOTS, device=dev)[None]
                    emb = emb.index_put((torch.arange(nq, device=dev)[:, None].expand_as(idx), idx),
                                        vis[0].expand(nq, -1, -1).to(emb.dtype))
                    h = model.encoder(inputs_embeds=emb, attention_mask=b["attention_mask"].to(dev)).last_hidden_state
                    h = h + model.type_emb(b["qtype"].to(dev))[:, None, :]
                    pad = ~b["attention_mask"].to(dev).bool()
                    for layer in model.head.layers:
                        h = layer(h, src_key_padding_mask=pad)
                    m = torch.gather(h, 1, b["marker_pos"].to(dev)[:, :, None].expand(-1, -1, h.size(-1)))
                    logits = model.scorer(m).squeeze(-1).float()
                    _ = logits.cpu()
                sync()
                if r >= 5:
                    times.append((time.perf_counter() - t0) * 1000)
            results["%s_q%d" % (name, nq)] = {"p50_ms": round(statistics.median(times), 1),
                                              "p95_ms": round(pctl(times, 95), 1)}
            print(name, nq, results["%s_q%d" % (name, nq)], flush=True)
        del tower
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if "so400m_q1" in results:
        results["decision"] = "so400m" if results["so400m_q1"]["p50_ms"] <= 100 else "b16"
    print("\nRESULT", json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
