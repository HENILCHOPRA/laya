"""Stage A alignment by image-conditioned masked language modelling (the design doc's planned retry).

Sequence: [CLS] <img slots> [SEP] caption [SEP]. 25% of caption tokens are masked and predicted
through mmBERT's MLM head (dense + norm from jhu-clsp/mmBERT-base, decoder tied to Laya's own
token embeddings). Every masked token is a training signal, far denser than one answer per
caption-matching row. By default only the vision adapter trains (the text encoder stays exactly
as shipped); --train-encoder also adapts the encoder at a low rate.

The eval reports masked-token accuracy with the real image and with images shuffled across the
batch: the gap is a direct measure of how much caption information the image tokens carry.

    python research/vision/train_mlm.py --base /workspace/ckpt/laya/multilingual --data /workspace/data \
        --out /workspace/runs/mlm --steps 3000
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
import vdata  # noqa: E402
from laya.vision import IMAGE_SLOTS, fit_feature_stats, load_image, pad_patches, patchify  # noqa: E402


class CapDataset(Dataset):
    def __init__(self, rows, tok, image_root, mask_rate=0.25, max_cap=64, seed=0):
        self.rows, self.tok, self.root = rows, tok, image_root
        self.mask_rate, self.max_cap, self.seed = mask_rate, max_cap, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(self.seed * 1000003 + i)
        img = load_image(os.path.join(self.root, r["image"]))
        budget = vdata.DOMAIN_BUDGET.get(r["domain"], 576)
        patches, shape = patchify(img, budget)
        cap = rng.choice(r["captions"])
        ids = self.tok(cap, add_special_tokens=False, truncation=True, max_length=self.max_cap)["input_ids"]
        tok = self.tok
        seq = [tok.cls_token_id] + [tok.pad_token_id] * IMAGE_SLOTS + [tok.sep_token_id]
        start = len(seq)
        seq += ids + [tok.sep_token_id]
        labels = [-100] * len(seq)
        n_mask = max(1, int(round(len(ids) * self.mask_rate)))
        for j in rng.sample(range(len(ids)), min(n_mask, len(ids))):
            labels[start + j] = seq[start + j]
            seq[start + j] = tok.mask_token_id
        return {"ids": seq, "labels": labels, "patches": patches, "shape": shape}


def collate(batch, pad_id):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        lab[i, : len(b["labels"])] = torch.tensor(b["labels"])
        att[i, : len(b["ids"])] = 1
    px = pad_patches([b["patches"] for b in batch], [b["shape"] for b in batch])
    return {"input_ids": ids, "labels": lab, "attention_mask": att, "pixels": px}


class MLMHead(nn.Module):
    """mmBERT's prediction head; the decoder weight is Laya's token-embedding matrix."""

    def __init__(self, embeddings: nn.Embedding, mlm_repo="jhu-clsp/mmBERT-base"):
        super().__init__()
        from transformers import AutoModelForMaskedLM
        ref = AutoModelForMaskedLM.from_pretrained(mlm_repo)
        self.head = ref.head
        self.bias = nn.Parameter(ref.decoder.bias.detach().clone())
        self.embeddings = embeddings
        del ref

    def forward(self, h):
        return nn.functional.linear(self.head(h), self.embeddings.weight, self.bias)


def forward_mlm(model, head, b, device, shuffle=False):
    px = {k: v.to(device, non_blocking=True) for k, v in b["pixels"].items()}
    vis = model.vision(**px)
    if shuffle:
        vis = vis.roll(1, dims=0)
    n = vis.size(0)
    ids = b["input_ids"].to(device, non_blocking=True)
    emb = model.embed_with_images(ids, vis, torch.arange(n, device=device), torch.ones(n, dtype=torch.long, device=device))
    h = model.encoder(inputs_embeds=emb, attention_mask=b["attention_mask"].to(device)).last_hidden_state
    lab = b["labels"].to(device)
    sel = lab != -100
    logits = head(h[sel])
    return nn.functional.cross_entropy(logits.float(), lab[sel]), (logits.argmax(-1) == lab[sel]).float().mean()


@torch.no_grad()
def evaluate(model, head, loader, device, max_batches=30):
    model.eval()
    out = {"real_loss": 0.0, "real_acc": 0.0, "shuf_loss": 0.0, "shuf_acc": 0.0}
    n = 0
    for bi, b in enumerate(loader):
        if bi >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l1, a1 = forward_mlm(model, head, b, device)
            l2, a2 = forward_mlm(model, head, b, device, shuffle=True)
        out["real_loss"] += l1.item(); out["real_acc"] += a1.item()
        out["shuf_loss"] += l2.item(); out["shuf_acc"] += a2.item()
        n += 1
    model.train()
    return {k: v / max(1, n) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="laya-multilingual checkpoint (fresh adapter)")
    ap.add_argument("--init", help="Laya-V checkpoint to continue from")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr-adapter", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--train-encoder", action="store_true")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-images", type=int, default=300000)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    if args.init:
        model, cfg = T.load_laya_v(args.init)
        tok = T.load_tokenizer(args.init)
    else:
        model, cfg = T.load_base(args.base, T.DEFAULT_VISION, device)
        tok = T.load_tokenizer(args.base)
    model.to(device)
    model.vision.tower.to(torch.bfloat16)
    head = MLMHead(model.encoder.get_input_embeddings()).to(device)

    rows = []
    for f in sorted(glob.glob(os.path.join(args.data, "items", "captions_*.jsonl"))):
        with open(f, encoding="utf-8") as g:
            rows += [json.loads(l) for l in g]
    held = vdata.holdout_keys(os.path.join(args.data, "items", "teacher_items.jsonl"))
    rows = [r for r in rows if r.get("captions") and os.path.exists(os.path.join(args.data, r["image"]))
            and not (r["split"] == "train" and r.get("srckey") in held)]
    train_rows = [r for r in rows if r["split"] == "train"]
    random.Random(0).shuffle(train_rows)
    train_rows = train_rows[: args.max_images]
    eval_rows = [r for r in rows if r["split"] != "train"]
    random.Random(1).shuffle(eval_rows)
    print("images train %d eval %d" % (len(train_rows), len(eval_rows)), flush=True)
    mk = lambda rs, seed: DataLoader(CapDataset(rs, tok, args.data, seed=seed), batch_size=args.batch, shuffle=True,
                                     num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True,
                                     collate_fn=lambda b: collate(b, tok.pad_token_id))
    loader, eval_loader = mk(train_rows, 0), mk(eval_rows[:4096], 1)

    for p in model.parameters():
        p.requires_grad_(False)
    adapter = [p for n, p in model.vision.named_parameters() if not n.startswith("tower.")]
    for p in adapter:
        p.requires_grad_(True)
    groups = [{"params": adapter, "lr": args.lr_adapter}]
    if args.train_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(True)
        groups.append({"params": list(model.encoder.parameters()), "lr": args.lr_encoder})
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.encoder.eval()
    head.requires_grad_(False)

    if not args.init:
        stats = []
        for bi, b in enumerate(loader):
            stats.append(b["pixels"])
            if len(stats) >= 20:
                break
        print("feature stats from %d patches" % fit_feature_stats(model.vision, stats, device), flush=True)

    opt = torch.optim.AdamW(groups, weight_decay=0.01, betas=(0.9, 0.98))
    base_lrs = [g["lr"] for g in opt.param_groups]
    print("before:", json.dumps(evaluate(model, head, eval_loader, device, 10)), flush=True)
    step, t0 = 0, time.time()
    log = open(os.path.join(args.out, "mlm_log.jsonl"), "a")
    while step < args.steps:
        for b in loader:
            scale = min(1.0, (step + 1) / 100) * max(0.05, 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, acc = forward_mlm(model, head, b, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "loss": loss.item(), "acc": acc.item(), "min": (time.time() - t0) / 60}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                ev = evaluate(model, head, eval_loader, device)
                print(json.dumps({"eval_step": step, **ev}), flush=True)
                log.write(json.dumps({"eval_step": step, **ev}) + "\n")
                log.flush()
            if step >= args.steps:
                break
    cfg["max_len"], cfg["head_max_len"] = cfg.get("max_len", 1024), cfg.get("head_max_len", 256)
    T.save_laya_v(model, cfg, tok, os.path.join(args.out, "final"), {"laya_v_stage": "A-mlm", "laya_v_step": step})
    print("done in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
