"""WiSE-FT for Laya-V: interpolate the text encoder and decision head between the base checkpoint
and the fine-tuned Laya-V checkpoint, keeping the fine-tuned vision adapter.

    merged = alpha * finetuned + (1 - alpha) * base      for every non-vision parameter

Recovers base text behaviour (smaller alpha) at some cost in image accuracy (larger alpha).

    python research/vision/merge_wise.py --base $CK --ft runs/stage_b2/final --alpha 0.7 --out runs/wise_07
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = load_file(os.path.join(args.base, "model.safetensors"))
    ft = load_file(os.path.join(args.ft, "model.safetensors"))
    out = {}
    for k, v in ft.items():
        if k.startswith("vision.") or k not in base:
            out[k] = v
        else:
            out[k] = (args.alpha * v.float() + (1 - args.alpha) * base[k].float()).to(v.dtype)
    shutil.rmtree(args.out, ignore_errors=True)
    shutil.copytree(args.ft, args.out, ignore=shutil.ignore_patterns("model.safetensors", "train_state.pt"))
    save_file(out, os.path.join(args.out, "model.safetensors"))
    cfg = json.load(open(os.path.join(args.out, "rl_agent_config.json")))
    cfg["wise_ft"] = {"alpha": args.alpha, "base": args.base, "finetuned": args.ft}
    json.dump(cfg, open(os.path.join(args.out, "rl_agent_config.json"), "w"), indent=2)
    print("merged", len(out), "tensors, alpha", args.alpha)


if __name__ == "__main__":
    main()
