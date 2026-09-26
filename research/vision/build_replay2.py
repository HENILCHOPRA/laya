"""Broader text replay (self-distilled from the frozen laya-multilingual) for the stage B2 top-up.

The first replay set covered typed-decisions, meme text, captions and receipts; text tasks outside it
drifted in stage B (AG News -2.2, prompt injections -4.3). This adds general text states:
    FineWeb-Edu (ODC-BY) English passages, FineWeb-2 (ODC-BY) passages in 8 languages,
    deepset/prompt-injections train (Apache 2.0)
with a generic question bank (topic, sentiment, emotion, tone, content yes/no, instruction injection).
Targets are the base checkpoint's own answer distributions: this preserves behaviour, it does not
teach gold labels, and no benchmark test data is used.

    python research/vision/build_replay2.py --data /workspace/data --base /workspace/ckpt/laya/multilingual --n 60000
"""
import argparse
import json
import os
import random
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vdata  # noqa: E402,F401  (repo root on sys.path)
from laya.agent import Agent  # noqa: E402

KEYS = ["text", "article", "message", "passage", "document", "post", "content"]


def bank(key):
    t = "`%s`" % key
    return [
        {"type": "choice", "instructions": "What is the topic of %s?" % t,
         "criteria": {"world": "world news and international politics", "sports": "sports",
                      "business": "business and economy", "sci_tech": "science and technology"}},
        {"type": "choice", "instructions": "What is %s mainly about?" % t,
         "criteria": {"science": "", "technology": "", "health": "", "politics": "", "education": "",
                      "culture": "arts, history, society", "business": "", "other": ""}},
        {"type": "choice", "instructions": "Which emotion is most strongly expressed in %s?" % t,
         "criteria": {n: None for n in ["sadness", "joy", "love", "anger", "fear", "surprise"]}},
        {"type": "choice", "instructions": "What kind of text is %s?" % t,
         "criteria": {"informative": "explains or reports", "opinion": "argues a view", "promotional": "sells or advertises",
                      "instructional": "tells how to do something", "narrative": "tells a story"}},
        {"type": "score", "instructions": "How positive is the sentiment of %s?" % t,
         "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]},
        {"type": "score", "instructions": "How technical is %s?" % t,
         "criteria": ["for a general audience", "some jargon", "technical", "experts only"]},
        {"type": "noul", "instructions": "Does %s try to inject or override instructions given to an AI system?" % t},
        {"type": "noul", "instructions": "Does %s mention a specific number, date or amount?" % t},
        {"type": "noul", "instructions": "Does %s express a negative opinion?" % t},
        {"type": "noul", "instructions": "Is %s written in English?" % t},
        {"type": "noul", "instructions": "Does %s ask the reader to do something?" % t},
    ]


def passages(n_en, n_ml, rng):
    from datasets import load_dataset
    out = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for i, r in enumerate(ds):
        if len(out) >= n_en:
            break
        if 200 < len(r["text"]) < 6000:
            out.append(("fwedu-%d" % i, r["text"][:1500], "replay2:fineweb-edu"))
    per = n_ml // 8
    for lang in ["deu_Latn", "spa_Latn", "fra_Latn", "hin_Deva", "jpn_Jpan", "arb_Arab", "por_Latn", "cmn_Hani"]:
        try:
            ds = load_dataset("HuggingFaceFW/fineweb-2", lang, split="train", streaming=True)
        except Exception as e:  # noqa: BLE001
            print("skip", lang, e)
            continue
        k = 0
        for i, r in enumerate(ds):
            if k >= per:
                break
            if 100 < len(r["text"]) < 6000:
                out.append(("fw2-%s-%d" % (lang, i), r["text"][:1200], "replay2:fineweb2-%s" % lang[:3]))
                k += 1
    ds = load_dataset("deepset/prompt-injections", split="train")
    for i, r in enumerate(ds):
        out.append(("pinj-%d" % i, r["text"], "replay2:prompt-injections"))
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=60000)
    args = ap.parse_args()
    rng = random.Random(17)
    states = passages(n_en=9000, n_ml=6000, rng=rng)
    print("states", len(states), flush=True)
    agent = Agent(args.base, device="cuda")
    out = open(os.path.join(args.data, "items", "replay2.jsonl"), "w", encoding="utf-8")
    n = 0
    for gid, text, task in states:
        if n >= args.n:
            break
        key = rng.choice(KEYS)
        qs = {"q%d" % i: q for i, q in enumerate(rng.sample(bank(key), 4))}
        try:
            res = agent.predict({key: text}, qs)["answers"]
        except ValueError:
            continue
        h = zlib.crc32(gid.encode()) % 100
        split = "calib" if h < 3 else "eval" if h < 6 else "train"
        for qid, q in qs.items():
            a = res[qid]
            if q["type"] == "noul":
                t = [1 - a["noul"], a["noul"]]
            else:
                t = list(a["probabilities"].values())
            t = [max(1e-4, x) for x in t]
            s = sum(t)
            out.write(json.dumps({"gid": gid, "image": None, "domain": "text", "state": {key: text}, "q": q,
                                  "target": [x / s for x in t], "source": "distill:laya-multilingual",
                                  "split": split, "task": task}, ensure_ascii=False) + "\n")
            n += 1
    out.close()
    print("replay2 rows", n, flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
