"""Label pool images with typed Laya questions using the OpenRouter teacher.

One request per image: the image plus 10 typed questions drawn from the question bank. The
teacher returns, for every question, a probability per option in the order shown (the model has
no log-probabilities, so stated probabilities are the soft targets). For ~15% of images a second
request shows the options in a shuffled order (the order check, merged in build_teacher_items.py).

Append-only JSONL output; rerunning resumes where it stopped.

    set OPENROUTER_API_KEY=...
    python research/vision/teacher_label.py --pool E:/LAYA-V/data/pool --out E:/LAYA-V/data/labels/teacher.jsonl --limit 30   # probe
    python research/vision/teacher_label.py --pool E:/LAYA-V/data/pool --out E:/LAYA-V/data/labels/teacher.jsonl --concurrency 96
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import question_bank as qbank  # noqa: E402
from openrouter import Client, image_part, message_json  # noqa: E402

N_QUESTIONS = 10
SHUFFLE_PCT = 15
PROB_FLOOR = 0.01
TRANSLATED_SHARE = 0.20
SOURCE = "teacher:space-bunny"

SYSTEM = """You label images for training a calibrated decision model.
For each numbered question, look carefully at the image and give a probability for EVERY option, in exactly the order the options are listed. Probabilities are numbers between 0 and 1 and must sum to 1 for each question.
Be calibrated: put more than 0.95 on one option only when the image makes the answer unambiguous. When the image is ambiguous or does not show enough, spread the probability to reflect your real uncertainty. Questions may be in any language; answer them all.
For yes/no questions the options are "no" (the statement does not hold for this image) and "yes" (it holds).
Return JSON only: {"answers": [{"qid": "q1", "probs": [...]}, ...]} with one entry per question."""

SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"qid": {"type": "string"},
                               "probs": {"type": "array", "items": {"type": "number"}}},
                "required": ["qid", "probs"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["answers"],
    "additionalProperties": False,
}


def stable_int(*parts) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def load_bank(path: Optional[str]) -> Dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def sample_questions(row: Dict, bank: Dict) -> List[Dict]:
    """Deterministic per image: the main and shuffled requests see the same questions."""
    domain = "meme" if row["domain"] in ("meme", "meme_base") else row["domain"]
    rng = random.Random(stable_int("questions", row["img_id"]))
    ts = qbank.TEMPLATES[domain]
    by_type = {t: [x for x in ts if x["type"] == t] for t in ("noul", "choice", "score")}
    n_score = 2 + (rng.random() < 0.5)
    n_choice = 3 + (rng.random() < 0.5)
    picks = (rng.sample(by_type["noul"], N_QUESTIONS - n_score - n_choice)
             + rng.sample(by_type["choice"], n_choice) + rng.sample(by_type["score"], n_score))
    rng.shuffle(picks)
    out = []
    for i, t in enumerate(picks):
        key = qbank.template_key(domain, t)
        ins, crit, lang, slot = t["instructions"], t.get("criteria"), "en", None
        entry = bank.get(key, {})
        tr = entry.get("translations", {})
        if tr and rng.random() < TRANSLATED_SHARE:
            lang = rng.choice(sorted(tr))
            ins = tr[lang]["instructions"]
            crit = tr[lang].get("criteria", crit)
        elif entry.get("paraphrases"):
            ins = rng.choice([t["instructions"]] + entry["paraphrases"])
        if "{obj}" in ins:
            present = row.get("meta", {}).get("objects", [])
            if present and rng.random() < 0.5:
                slot = rng.choice(present)
            else:
                slot = rng.choice([o for o in qbank.COCO_OBJECTS if o not in present])
            ins = ins.replace("{obj}", slot)
        q = {"qid": "q%d" % (i + 1), "tkey": key, "lang": lang, "type": t["type"], "instructions": ins}
        if crit is not None:
            q["criteria"] = crit
        if slot:
            q["slot"] = slot
        out.append(q)
    return out


def option_texts(q: Dict) -> List[str]:
    if q["type"] == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in q["criteria"].items()]
    if q["type"] == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(q["criteria"])]
    return ["no", "yes"]


def render_prompt(questions: List[Dict], orders: Dict[str, List[int]]) -> str:
    lines = ["Questions about the image:"]
    for q in questions:
        opts = option_texts(q)
        kind = {"noul": "yes/no", "choice": "choose one", "score": "rate on the scale"}[q["type"]]
        lines.append("\n%s (%s): %s" % (q["qid"], kind, q["instructions"]))
        for j, oi in enumerate(orders[q["qid"]]):
            lines.append("  option %d. %s" % (j + 1, opts[oi]))
    lines.append("\nReturn one probability per option, in the listed order, for every question.")
    return "\n".join(lines)


def floor_renorm(p: List[float]) -> List[float]:
    p = [max(PROB_FLOOR, x) for x in p]
    s = sum(p)
    return [x / s for x in p]


def validate(parsed, questions, orders, allow_onehot=False):
    """Returns {qid: probs in label order} or an error string. `allow_onehot` keeps all-confident answers
    (rejected when labelling, since they carry no calibration signal; kept when scoring a model)."""
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answers"), list):
        return "schema"
    want = {q["qid"]: q for q in questions}
    got = {}
    for a in parsed["answers"]:
        if not isinstance(a, dict) or a.get("qid") not in want:
            return "unknown qid %r" % (a.get("qid") if isinstance(a, dict) else a)
        q = want[a["qid"]]
        probs = a.get("probs")
        k = len(orders[q["qid"]])
        if not isinstance(probs, list) or len(probs) != k:
            return "wrong option count for %s" % q["qid"]
        try:
            probs = [float(x) for x in probs]
        except (TypeError, ValueError):
            return "non-numeric"
        if any((not math.isfinite(x)) or x < 0 for x in probs):
            return "bad value"
        s = sum(probs)
        if not (0.8 <= s <= 1.2):
            return "sum %.3f" % s
        probs = [x / s for x in probs]
        label_order = [0.0] * k
        for j, oi in enumerate(orders[q["qid"]]):
            label_order[oi] = probs[j]
        got[q["qid"]] = floor_renorm(label_order)
    if set(got) != set(want):
        return "missing %s" % sorted(set(want) - set(got))
    if not allow_onehot and all(max(v) >= 0.985 for v in got.values()):
        return "degenerate: all one-hot"
    return got


def orders_for(questions, img_id, variant):
    orders = {}
    for q in questions:
        k = len(option_texts(q))
        idx = list(range(k))
        if variant == "shuffle":
            rng = random.Random(stable_int("order", img_id, q["qid"]))
            while k > 1 and idx == list(range(k)):
                rng.shuffle(idx)
        orders[q["qid"]] = idx
    return orders


async def label_one(client, row, pool_dir, bank, variant, out_f, lock, counters):
    questions = sample_questions(row, bank)
    orders = orders_for(questions, row["img_id"], variant)
    content = [{"type": "text", "text": render_prompt(questions, orders)},
               image_part(os.path.join(pool_dir, row["path"]))]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    err = None
    for attempt in range(2):  # the design doc's "retry once" on invalid replies
        data = await client.chat(messages, schema=SCHEMA)
        if data is None:
            err = "no response"
            break
        res = validate(message_json(data), questions, orders)
        if isinstance(res, dict):
            rec = {"img_id": row["img_id"], "domain": row["domain"], "variant": variant,
                   "model": data.get("model", client.model), "source": SOURCE, "time": time.time(),
                   "usage": data.get("usage"), "questions": []}
            for q in questions:
                qq = dict(q)
                qq["order"] = orders[q["qid"]]
                qq["probs"] = res[q["qid"]]
                rec["questions"].append(qq)
            async with lock:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            counters["ok"] += 1
            return
        err = res
    counters["fail"] += 1
    counters.setdefault("errors", {}).setdefault(str(err)[:40], 0)
    counters["errors"][str(err)[:40]] += 1


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["img_id"], r["variant"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def summarize(path):
    maxes, by_dom, n = [], {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n += 1
            by_dom[r["domain"]] = by_dom.get(r["domain"], 0) + 1
            maxes += [max(q["probs"]) for q in r["questions"]]
    if not maxes:
        return
    maxes.sort()
    print("records %d | by domain %s" % (n, by_dom))
    print("max-prob: mean %.3f | p10 %.3f | p50 %.3f | share>=0.95 %.2f | share<0.6 %.2f" % (
        statistics.mean(maxes), maxes[len(maxes) // 10], maxes[len(maxes) // 2],
        sum(m >= 0.95 for m in maxes) / len(maxes), sum(m < 0.6 for m in maxes) / len(maxes)))


async def main_async(args):
    pool = []
    with open(os.path.join(args.pool, "pool.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["domain"] == "meme_base":  # memes are labelled from make_memes.py output
                continue
            if args.domains and r["domain"] not in args.domains:
                continue
            pool.append(r)
    # interleave domains so a partial run stays balanced
    pool.sort(key=lambda r: stable_int("sched", r["img_id"]))
    jobs = []
    for r in pool:
        jobs.append((r, "main"))
        if stable_int("shuffle", r["img_id"]) % 100 < SHUFFLE_PCT:
            jobs.append((r, "shuffle"))
    done = load_done(args.out)
    jobs = [j for j in jobs if (j[0]["img_id"], j[1]) not in done]
    if args.limit:
        jobs = jobs[:args.limit]
    print("jobs to run: %d (already done: %d)" % (len(jobs), len(done)))
    bank = load_bank(args.bank)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    client = Client(concurrency=args.concurrency)
    counters = {"ok": 0, "fail": 0}
    lock = asyncio.Lock()
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as out_f:
        async def reporter():
            while True:
                await asyncio.sleep(60)
                lat = client.stats["latency"][-500:]
                print("[%5.1f min] ok %d fail %d | http ok %d err %d 429 %d | p50 latency %.1fs | %s" % (
                    (time.time() - t0) / 60, counters["ok"], counters["fail"], client.stats["ok"],
                    client.stats["http_err"], client.stats["429"],
                    statistics.median(lat) if lat else float("nan"), counters.get("errors", {})), flush=True)

        rep = asyncio.create_task(reporter())
        # A fixed worker pool: one slow request never holds up the others, and at most
        # `concurrency` base64 images are in memory at once.
        queue: asyncio.Queue = asyncio.Queue()
        for j in jobs:
            queue.put_nowait(j)

        async def worker():
            while True:
                try:
                    r, v = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    await label_one(client, r, args.pool, bank, v, out_f, lock, counters)
                except SystemExit:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad image must not stop the run
                    counters["fail"] += 1
                    print("job error", r["img_id"], repr(e)[:200])

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        rep.cancel()
    await client.close()
    lat = client.stats["latency"]
    print("done: ok %d fail %d in %.1f min; latency p50 %.1fs p90 %.1fs; errors %s" % (
        counters["ok"], counters["fail"], (time.time() - t0) / 60,
        statistics.median(lat) if lat else float("nan"),
        sorted(lat)[int(len(lat) * 0.9)] if lat else float("nan"), counters.get("errors", {})))
    summarize(args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bank", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_bank.json"))
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--domains", nargs="*")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
