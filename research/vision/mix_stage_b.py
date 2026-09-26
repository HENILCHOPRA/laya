"""Freeze the stage B mixture: teacher + public + text replay, with per-task caps.

Writes $DATA/items/stage_b.jsonl (train + calib + eval rows) and stage_b_manifest.json (source,
license, rows per split and share of training rows) for the model card.

    python research/vision/mix_stage_b.py --data /workspace/data
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

TRAIN_CAP = {  # per task prefix, question rows
    "vqav2": 40000, "aokvqa": 20000, "pixmo": 24000, "aria": 16000, "cord": 6000,
}
OVERSAMPLE = {"cord": 3}  # 800 receipts: repeat their rows
EVAL_CAP_PER_TASK = 2500
LICENSES = {"teacher:space-bunny": "teacher labels (Space Bunny Alpha, OpenRouter stealth; see model card)",
            "public:coco2017": "CC-BY-4.0", "public:vqav2": "CC-BY-4.0", "public:a-okvqa": "Apache-2.0",
            "public:pixmo-docs": "ODC-BY", "public:pixmo-cap": "ODC-BY-1.0", "public:cord-v2": "CC-BY-4.0",
            "public:aria-ui": "Apache-2.0", "public:screenspot": "Apache-2.0", "public:aria": "Apache-2.0",
            "distill:laya-multilingual": "Apache-2.0 (self-distillation of the base model)"}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    d = os.path.join(args.data, "items")
    rng = random.Random(11)
    rows = load(os.path.join(d, "teacher_items.jsonl"))
    for f in sorted(os.listdir(d)):
        if f.startswith("public_") and f.endswith(".jsonl"):
            rows += load(os.path.join(d, f))
    if os.path.exists(os.path.join(d, "replay.jsonl")):
        rows += load(os.path.join(d, "replay.jsonl"))

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vdata
    held = vdata.holdout_keys(os.path.join(d, "teacher_items.jsonl"))
    n0 = len(rows)
    rows = [r for r in rows if not (r["split"] == "train" and r.get("srckey") in held)]
    print("dropped %d training rows on held-out teacher images" % (n0 - len(rows)))
    by = defaultdict(list)
    for r in rows:
        by[(r["split"], r["task"].split(":")[0])].append(r)
    out = []
    for (split, prefix), rs in sorted(by.items()):
        # cap by groups (images), not rows, so an image keeps all of its questions together
        groups = defaultdict(list)
        for r in rs:
            groups[r["gid"]].append(r)
        gids = sorted(groups)
        rng.shuffle(gids)
        cap = TRAIN_CAP.get(prefix) if split == "train" else EVAL_CAP_PER_TASK if split == "eval" else None
        taken = []
        for g in gids:
            if cap is not None and len(taken) >= cap:
                break
            taken += groups[g]
        if split == "train":
            taken = taken * OVERSAMPLE.get(prefix, 1)
        out += taken
    with open(os.path.join(d, "stage_b.jsonl"), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["source"], r["split"]) for r in out)
    n_train = sum(v for (s, sp), v in c.items() if sp == "train")
    man = {"rows": {"%s|%s" % k: v for k, v in sorted(c.items())},
           "train_share": {s: round(sum(v for (s2, sp), v in c.items() if s2 == s and sp == "train") / n_train, 4)
                           for s in sorted({s for s, _ in c})},
           "licenses": {s: LICENSES.get(s, "?") for s in sorted({s for s, _ in c})}}
    with open(os.path.join(d, "stage_b_manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
