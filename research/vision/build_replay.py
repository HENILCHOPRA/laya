"""Text replay rows for stage B, self-distilled from the frozen laya-multilingual (A100).

Targets are the base checkpoint's own answer distributions, so replay teaches "answer text-only
states exactly as before", which is what the text regression check measures. Sources carry
licenses compatible with Apache 2.0 and none overlaps the regression benchmarks:
    typed-decisions train (Apache 2.0)             states + questions as published
    synthetic meme / post texts (teacher-written)  text state, meme-domain question bank
    COCO captions (CC-BY 4.0)                      caption as text state, photo-domain questions
    CORD receipts (CC-BY 4.0) as JSON              receipt-domain questions

    python research/vision/build_replay.py --data /workspace/data --raw /workspace/raw \
        --base /workspace/ckpt/laya/multilingual --n 40000
"""
import argparse
import glob
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import question_bank as qbank  # noqa: E402
import vdata  # noqa: E402,F401  (repo root on sys.path)
from laya.agent import Agent  # noqa: E402

TEXT_OK = {  # template ids that still make sense when the "image" is described in text
    "photo": ["people_present", "animal_present", "outdoors", "vehicle_present", "food_present", "night",
              "group_of_people", "sport_activity", "water_present", "setting", "main_subject", "activity",
              "animal_type", "people_count", "crowd_density", "safety_hazard", "formality"],
    "meme": ["humorous", "offensive", "sarcastic", "political", "promotional", "targets_person", "profanity",
             "call_to_action", "factual_claim", "sentiment", "emotion", "humor_type", "topic", "intent",
             "moderation", "toxicity", "funniness", "political_intensity"],
    "receipt": ["many_items", "has_tax", "has_discount", "paid_card", "alcohol", "category", "payment_method",
                "item_count", "merchant_type", "expense_category", "line_items", "order_size", "detail_level"],
}


def questions_for(domain, rng, k):
    ts = {t["id"]: t for t in qbank.TEMPLATES[domain]}
    out = {}
    for tid in rng.sample(TEXT_OK[domain], min(k, len(TEXT_OK[domain]))):
        t = ts[tid]
        q = {"type": t["type"], "instructions": t["instructions"]}
        if "criteria" in t:
            q["criteria"] = t["criteria"]
        out[tid] = q
    return out


def states(args, rng):
    """Yields (gid, state, questions, task)."""
    from datasets import load_dataset
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="train")
    for i, row in enumerate(ds):
        yield "td-%d" % i, json.loads(row["state"]), json.loads(row["questions"]), "replay:typed-decisions"
    with open(os.path.join(args.data, "pool", "pool.meme.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = r["meta"]["text"]
            text = (t.get("top", "") + " " + t.get("bottom", "")).strip()
            yield "txt-" + r["img_id"], {"post": text}, questions_for("meme", rng, 6), "replay:meme-text"
    caps = json.load(open(os.path.join(args.data, "coco", "annotations", "captions_train2017.json")))["annotations"]
    rng.shuffle(caps)
    for a in caps[:5000]:
        yield "txt-coco-%d" % a["id"], {"description": a["caption"].strip()}, questions_for("photo", rng, 3), \
            "replay:caption-text"
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(os.path.join(args.raw, "cord", "data", "train-*.parquet"))):
        for i, r in enumerate(pq.read_table(f, columns=["ground_truth"]).to_pylist()):
            gt = json.loads(r["ground_truth"]).get("gt_parse", {})
            yield "txt-cord-%s-%d" % (os.path.basename(f)[:14], i), {"receipt": gt}, questions_for("receipt", rng, 6), \
                "replay:receipt-json"


def target_of(q, ans):
    if q["type"] == "noul":
        p = ans["noul"]
        return [1 - p, p]
    if q["type"] == "choice":
        return [ans["probabilities"][k] for k in q["criteria"]]
    return [ans["probabilities"][str(i)] for i in range(len(q["criteria"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=40000)
    args = ap.parse_args()
    rng = random.Random(7)
    agent = Agent(args.base, device="cuda")
    out = open(os.path.join(args.data, "items", "replay.jsonl"), "w", encoding="utf-8")
    n = 0
    for gid, state, qs, task in states(args, rng):
        if n >= args.n:
            break
        try:
            res = agent.predict(state, qs)["answers"]
        except ValueError:
            continue
        h = zlib.crc32(gid.encode()) % 100
        split = "calib" if h < 3 else "eval" if h < 6 else "train"
        for qid, q in qs.items():
            t = [max(1e-4, x) for x in target_of(q, res[qid])]
            s = sum(t)
            out.write(json.dumps({"gid": gid, "image": None, "domain": "text", "state": state, "q": q,
                                  "target": [x / s for x in t], "source": "distill:laya-multilingual",
                                  "split": split, "task": task}, ensure_ascii=False) + "\n")
            n += 1
    out.close()
    print("replay rows", n, flush=True)


if __name__ == "__main__":
    main()
