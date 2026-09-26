"""Answer the blind audit with a strong LLM judge instead of a human (same output as audit.html's export).

Reads the items straight from audit.html (the exact images and options a human would see) plus any state
text from audit_items.jsonl, asks the judge to pick the correct option or abstain, and writes
audit_verdicts.json in the format eval_vision.py --audit expects. The teacher's answer is never shown.
Each verdict carries "judge" so the report can say the reference is an LLM, not a human.

    set OPENROUTER_API_KEY=...
    python research/vision/judge_audit.py --page E:/LAYA-V/data/audit.html \
        --items E:/LAYA-V/data/items/audit_items.jsonl --out audit_verdicts.json
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openrouter import Client, message_json  # noqa: E402

JUDGE = "anthropic/claude-sonnet-5"

PROMPT = """You are checking labels for an image-understanding dataset. Look carefully at the image and answer the question by choosing exactly one of the numbered options.

Question: {q}
{state}Options:
{opts}

Rules:
- Pick the option that is correct for this image. If several are plausible, pick the best one.
- Answer null only if the question cannot be answered from the image or no option is reasonable (a bad question).
- Reply with JSON only: {{"answer": <option number or null>, "reason": "<one short sentence>"}}"""


def load_page(path):
    s = open(path, encoding="utf-8").read()
    i = s.index("const D=") + len("const D=")
    j = s.index(";const KEY=", i)
    return json.loads(s[i:j])


async def judge_one(client, d, state, n_opts):
    opts = "\n".join("%d. %s" % (k + 1, o) for k, o in enumerate(d["opts"]))
    st = ("Context given with the image: %s\n" % state) if state else ""
    msg = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + d["img"]}},
        {"type": "text", "text": PROMPT.format(q=d["q"], state=st, opts=opts)}]}]
    for _ in range(3):
        data = await client.chat(msg, reasoning_effort="medium", max_tokens=8000, temperature=0.0)
        if data is None:
            continue
        js = message_json(data)
        if not isinstance(js, dict) or "answer" not in js:
            continue
        a = js["answer"]
        if a is None:
            return None, js.get("reason", "")
        if isinstance(a, str) and a.strip().isdigit():
            a = int(a.strip())
        if isinstance(a, int) and 1 <= a <= n_opts:
            return a - 1, js.get("reason", "")
    return "failed", ""


async def run(args):
    data = load_page(args.page)
    states = {}
    if args.items and os.path.exists(args.items):
        for i, line in enumerate(open(args.items, encoding="utf-8")):
            st = json.loads(line).get("state") or ""
            states[i] = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    done = {}
    if os.path.exists(args.out + ".partial"):
        for line in open(args.out + ".partial", encoding="utf-8"):
            r = json.loads(line)
            done[r["i"]] = r
    todo = [d for d in data if d["i"] not in done]
    print("items %d, already judged %d, to do %d" % (len(data), len(done), len(todo)), flush=True)
    client = Client(concurrency=args.concurrency, model=args.model)
    part = open(args.out + ".partial", "a", encoding="utf-8")

    async def one(d):
        ans, reason = await judge_one(client, d, states.get(d["i"], ""), len(d["opts"]))
        if ans == "failed":
            print("failed item", d["i"], flush=True)
            return
        r = {"i": d["i"], "human": ans, "judge": args.model, "reason": reason}
        done[d["i"]] = r
        part.write(json.dumps(r, ensure_ascii=False) + "\n")
        part.flush()
        if len(done) % 25 == 0:
            print("judged", len(done), flush=True)

    await asyncio.gather(*(one(d) for d in todo))
    await client.close()
    part.close()
    out = [done[i] for i in sorted(done)]
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    n_null = sum(r["human"] is None for r in out)
    print("wrote %s: %d verdicts (%d abstained as bad/unsure), %d missing" % (
        args.out, len(out), n_null, len(data) - len(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", required=True, help="audit.html built by build_audit_page.py")
    ap.add_argument("--items", help="audit_items.jsonl (adds any state text shown with the image)")
    ap.add_argument("--out", default="audit_verdicts.json")
    ap.add_argument("--model", default=JUDGE)
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
