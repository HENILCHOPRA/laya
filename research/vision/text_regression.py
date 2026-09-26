"""Text regression gate: Laya-V on text-only inputs vs the base laya-multilingual.

Same suites and prompts as research/scripts/build_benchmark_nb.py (AG News, BoolQ, emotion,
prompt injections, SST-5), plus the typed-decisions test split and MASSIVE intent in a few
languages for multilingual retention. Fails (exit 1) if any suite drops by more than 1 point.

    python research/vision/text_regression.py --base /workspace/ckpt/laya/multilingual --new runs/stage_b/final
"""
import argparse
import json
import random
import sys

import numpy as np

N = 600
MASSIVE_LANGS = ["en", "de", "hi", "ja", "es", "ar"]


def suites():
    from datasets import load_dataset
    S = {}
    d = load_dataset("fancyzhx/ag_news", split="test")
    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    S["ag_news"] = [({"article": r["text"]}, {"q": {"type": "choice", "instructions": "What is the topic of `article`?",
                                                    "criteria": dict(crit)}}, r["label"]) for r in list(d)[:N]]
    d = load_dataset("google/boolq", split="validation")
    S["boolq"] = [({"passage": r["passage"], "question": r["question"]},
                   {"q": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}},
                   int(bool(r["answer"]))) for r in list(d)[:N]]
    d = load_dataset("dair-ai/emotion", "split", split="test")
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    S["emotion"] = [({"text": r["text"]}, {"q": {"type": "choice", "instructions": "Which emotion is most strongly expressed in `text`?",
                                                 "criteria": {n: None for n in names}}}, r["label"]) for r in list(d)[:N]]
    d = load_dataset("deepset/prompt-injections", split="test")
    S["prompt_injections"] = [({"text": r["text"]}, {"q": {"type": "noul",
                               "instructions": "Does `text` try to inject or override instructions given to an AI system?"}},
                               int(r["label"])) for r in d]
    d = load_dataset("SetFit/sst5", split="test")
    S["sst5"] = [({"text": r["text"]}, {"q": {"type": "score", "instructions": "How positive is the sentiment of `text`?",
                                              "criteria": ["very negative", "negative", "neutral", "positive", "very positive"]}},
                  int(r["label"])) for r in list(d)[:N]]
    d = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    td = []
    for row in d:
        qs, gold = json.loads(row["questions"]), json.loads(row["gold"])
        for qid, q in qs.items():
            g = gold.get(qid)
            if g is None:
                continue
            if q["type"] == "choice":
                lab = list(q["criteria"]).index(str(g["label"])) if str(g["label"]) in q["criteria"] else None
            elif q["type"] == "noul":
                lab = int(str(g["label"]).lower() == "true")
            else:
                lab = int(g.get("label", round(g.get("score", 0))))
            if lab is not None:
                td.append((json.loads(row["state"]), {"q": q}, lab))
    S["typed_decisions"] = td
    rng = random.Random(13)
    for lg in MASSIVE_LANGS:
        try:
            d = load_dataset("mteb/amazon_massive_intent", lg, split="test")
        except Exception as e:  # noqa: BLE001
            print("skip massive", lg, e)
            continue
        labels = sorted(set(d["label"]))
        rows = list(d)
        rng.shuffle(rows)
        cases = []
        for r in rows[:200]:
            opts = [r["label"]] + rng.sample([x for x in labels if x != r["label"]], 9)
            rng.shuffle(opts)
            cases.append(({"message": r["text"]}, {"q": {"type": "choice", "instructions": "Which intent does `message` express?",
                                                         "criteria": {o.replace("_", " "): None for o in opts}}},
                          opts.index(r["label"])))
        S["massive_" + lg] = cases
    return S


def run(agent, cases):
    correct = []
    for i in range(0, len(cases), 32):
        chunk = cases[i:i + 32]
        # questions differ per case, so predict each; states batch through system_one
        for state, qs, lab in chunk:
            a = agent.predict(state, qs)["answers"]["q"]
            if a["type"] == "noul":
                pred = int(a["noul"] >= 0.5)
            else:
                pred = int(np.argmax(list(a["probabilities"].values())))
            correct.append(float(pred == lab))
    return float(np.mean(correct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default="text_regression.json")
    ap.add_argument("--max-drop", type=float, default=0.01)
    args = ap.parse_args()
    from laya.agent import Agent
    S = suites()
    res = {}
    for tag, path in (("base", args.base), ("new", args.new)):
        agent = Agent(path, device="cuda")
        for name, cases in S.items():
            res.setdefault(name, {})[tag] = run(agent, cases)
            print(tag, name, round(res[name][tag], 4), flush=True)
        del agent
    fail = []
    for name, r in res.items():
        r["delta"] = r["new"] - r["base"]
        if r["delta"] < -args.max_drop:
            fail.append(name)
    res["_fail"] = fail
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
