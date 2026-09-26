"""Stage C: fit calibration temperatures on the held-out calib slice, per modality.

Text rows fit `temperature` (per type) and `temperature_by_options` buckets as before; image rows
fit `temperature_image` (per type) and `image|<bucket>` entries. LBFGS on soft cross-entropy,
exactly as the fine-tuning notebook's `fit_one_temp`. Writes the values into the checkpoint's
rl_agent_config.json and reports ECE before/after on the eval split.

    python research/vision/calibrate.py --ckpt runs/stage_b/final --items data/items/stage_b.jsonl \
        --image-root data
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
import vdata  # noqa: E402
from laya.common import TEMP_MAX, TEMP_MIN, ece_score, temp_bucket, temp_key  # noqa: E402

MIN_BUCKET = 150


def fit_one_temp(sel):
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    Tt = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        Tt[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(Tt * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), TEMP_MIN, TEMP_MAX).item())


def collect(model, items, tok, args, split):
    rows = [r for r in items if r["split"] == split]
    groups = vdata.group_items(rows)
    loader, _ = T.make_loader(groups, tok, args, shuffle=False)
    _, out = T.evaluate(model, loader, torch.device("cuda"), None, return_rows=True)
    return out


def temp_for(cfg, qt, k, image):
    tbo = cfg.get("temperature_by_options", {})
    t = tbo.get(temp_bucket(qt, k), cfg.get("temperature", [1, 1, 1])[qt])
    if image:
        ti = cfg.get("temperature_image")
        t = tbo.get(temp_key(qt, k, "image"), ti[qt] if ti else t)
    return t


def ece_of(rows, cfg):
    confs, correct = defaultdict(list), defaultdict(list)
    for r in rows:
        z = np.array(r["logits"])
        img = r["meta"]["domain"] != "text"
        p = vdata.softmax_np(z / temp_for(cfg, r["qtype"], len(z), img))
        t = np.array(r["target"])
        for key in ("all", "image" if img else "text"):
            confs[key].append(p.max())
            correct[key].append(float(p.argmax() == t.argmax()))
    return {k: round(ece_score(np.array(confs[k]), np.array(correct[k])), 4) for k in confs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()
    model, cfg = T.load_laya_v(args.ckpt)
    tok = T.load_tokenizer(args.ckpt)
    model.to("cuda").eval()
    model.vision.tower.to(torch.bfloat16)
    items = vdata.read_items(args.items)
    calib = collect(model, items, tok, args, "calib")
    evals = collect(model, items, tok, args, "eval")
    before = ece_of(evals, cfg)

    by_type, by_bucket = defaultdict(list), defaultdict(list)
    for r in calib:
        img = r["meta"]["domain"] != "text"
        z, t, qt = r["logits"], r["target"], r["qtype"]
        by_type[(img, qt)].append((z, t))
        key = temp_key(qt, len(z), "image") if img else temp_bucket(qt, len(z))
        by_bucket[key].append((z, t))
    new = dict(cfg)
    new["temperature"] = [fit_one_temp(by_type[(False, q)]) if len(by_type[(False, q)]) >= 30 else 1.0 for q in range(3)]
    new["temperature_image"] = [fit_one_temp(by_type[(True, q)]) if len(by_type[(True, q)]) >= 30 else 1.0 for q in range(3)]
    new["temperature_by_options"] = {k: fit_one_temp(v) for k, v in sorted(by_bucket.items()) if len(v) >= MIN_BUCKET}
    after = ece_of(evals, new)
    new["calibration"] = {"calib_rows": len(calib), "eval_rows": len(evals), "ece_before": before, "ece_after": after}
    with open(os.path.join(args.ckpt, "rl_agent_config.json"), "w") as f:
        json.dump(new, f, indent=2)
    print(json.dumps({"temperature": new["temperature"], "temperature_image": new["temperature_image"],
                      "temperature_by_options": new["temperature_by_options"], "ece_before": before,
                      "ece_after": after}, indent=1))


if __name__ == "__main__":
    main()
