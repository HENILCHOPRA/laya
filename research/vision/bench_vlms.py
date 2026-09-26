"""Laya-V against general VLMs served on OpenRouter: same images, same typed questions, same option order.

Every VLM gets the teacher's labelling prompt (a JSON probability per option, <= 10 questions per call,
temperature 0, reasoning effort low). Laya-V answers the same rows on the local GPU. Reported per system:
accuracy by slice, macro accuracy over the public gold task families with a 95% bootstrap CI (images
resampled), McNemar against Laya-V on paired questions, raw and temperature-fitted ECE, $ per 1,000
decisions, and latency (API: sequential 1 image + 1 question calls, network included).

    python research/vision/bench_vlms.py --items data/items/stage_b.jsonl --image-root data \
        --final /workspace/ckpt/laya-v --audit-items data/items/audit_items.jsonl \
        --audit-verdicts data/items/audit_verdicts_sonnet5.json --out runs/bench

Without a GPU, pass --laya-rows runs/bench/laya_v_rows.json from an earlier run instead of --final.
Raw API answers are cached under <out>/cache/, so an interrupted run resumes without paying twice.
"""
import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vdata  # noqa: E402
from eval_vision import metrics, select_groups, vlm_rows  # noqa: E402

MODELS = ["stealth/space-bunny-alpha", "anthropic/claude-sonnet-5", "openai/gpt-6-sol", "google/gemini-3.8-flash",
          "qwen/qwen3-vl-8b-instruct", "openai/gpt-6-luna"]
FAMILIES = ("vqav2", "aokvqa", "pixmo", "cord", "screenspot")


def family(task):
    return task.split(":")[0]


def safe(name):
    return name.replace("/", "__").replace(":", "_")


# ----------------------------------------------------------------------------- statistics
def correct_of(r):
    return float(int(np.argmax(r["p"])) == int(np.argmax(r["target"])) and not r.get("failed"))


def macro_gold(rows):
    """Mean over the public task families of the family's accuracy (so ScreenSpot's 8 slices count once)."""
    by = defaultdict(list)
    for r in rows:
        if family(r["task"]) in FAMILIES:
            by[family(r["task"])].append(correct_of(r))
    return float(np.mean([np.mean(v) for v in by.values()])) if by else float("nan"), sorted(by)


def bootstrap_macro(rows, n=1000, seed=0):
    """95% CI of macro_gold, resampling images within each family."""
    rng = np.random.default_rng(seed)
    fam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if family(r["task"]) in FAMILIES:
            fam[family(r["task"])][r["gid"]].append(correct_of(r))
    per = {f: (np.array([sum(v) for v in g.values()]), np.array([len(v) for v in g.values()])) for f, g in fam.items()}
    stats = []
    for _ in range(n):
        accs = []
        for s, c in per.values():
            idx = rng.integers(0, len(s), len(s))
            accs.append(s[idx].sum() / c[idx].sum())
        stats.append(np.mean(accs))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def mcnemar(a_rows, b_rows):
    """Exact two-sided McNemar test on questions both systems answered (paired by qkey)."""
    a = {r["qkey"]: correct_of(r) for r in a_rows}
    b = {r["qkey"]: correct_of(r) for r in b_rows}
    keys = set(a) & set(b)
    n01 = sum(1 for k in keys if a[k] and not b[k])
    n10 = sum(1 for k in keys if b[k] and not a[k])
    n = n01 + n10
    if n == 0:
        return {"n": len(keys), "a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n": len(keys), "a_only": n01, "b_only": n10, "p": p}


def ece(rows, bins=15):
    conf = np.array([max(r["p"]) for r in rows])
    ok = np.array([correct_of(r) for r in rows])
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def fitted_ece(rows):
    """ECE after one temperature on log-probabilities, fitted 2-fold by image (grid search on NLL)."""
    live = [r for r in rows if not r.get("failed")]
    if len(live) < 20:
        return None, None
    gids = sorted({r["gid"] for r in live})
    random.Random(0).shuffle(gids)
    fold = {g: i % 2 for i, g in enumerate(gids)}
    grid = np.exp(np.linspace(math.log(0.2), math.log(10), 120))

    def scale(r, t):
        z = np.log(np.clip(np.array(r["p"]), 1e-6, 1)) / t
        z = np.exp(z - z.max())
        return z / z.sum()

    def nll(rs, t):
        return -np.mean([math.log(max(scale(r, t)[int(np.argmax(r["target"]))], 1e-12)) for r in rs])

    out, temps = [], []
    for f in (0, 1):
        fit = [r for r in live if fold[r["gid"]] == f]
        app = [r for r in live if fold[r["gid"]] != f]
        t = float(min(grid, key=lambda x: nll(fit, x)))
        temps.append(t)
        out += [dict(r, p=scale(r, t).tolist()) for r in app]
    return ece(out + [r for r in rows if r.get("failed")]), temps


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


def summarize(rows, audit_rows=None, laya_rows=None):
    gold = [r for r in rows if family(r["task"]) in FAMILIES]
    teach = [r for r in rows if r["task"].startswith("teacher:")]
    s = {"metrics": metrics(rows), "n_rows": len(rows), "failed_rows": sum(bool(r.get("failed")) for r in rows)}
    s["macro_gold"], s["families"] = macro_gold(gold)
    s["macro_gold_ci95"] = bootstrap_macro(gold) if gold else None
    s["gold_acc"] = float(np.mean([correct_of(r) for r in gold])) if gold else None
    s["teacher_agreement"] = float(np.mean([correct_of(r) for r in teach])) if teach else None
    s["ece_raw"] = ece(rows) if rows else None
    s["ece_fitted"], s["temperatures"] = fitted_ece(rows)
    if audit_rows:
        s["audit_acc"] = float(np.mean([correct_of(r) for r in audit_rows]))
        s["audit_n"] = len(audit_rows)
    if laya_rows is not None and laya_rows is not rows:
        s["mcnemar_vs_laya_v_gold"] = mcnemar([r for r in laya_rows if family(r["task"]) in FAMILIES], gold)
    return s


# ----------------------------------------------------------------------------- latency
def api_latency(model, groups, image_root, n=50, seed=0):
    """Sequential 1 image + 1 question calls; wall time per call (network included)."""
    import teacher_label as TL
    from openrouter import Client, image_part, message_json

    rng = random.Random(seed)
    picks = rng.sample([g for g in groups if g[0].get("image")], min(n, len(groups)))

    async def go():
        client = Client(concurrency=1, model=model)
        times, ok = [], 0
        for g in picks:
            r = g[0]
            q = {"qid": "q1", "type": r["q"]["type"], "instructions": r["q"]["instructions"]}
            if "criteria" in r["q"]:
                q["criteria"] = r["q"]["criteria"]
            orders = {"q1": list(range(len(TL.option_texts(q))))}
            msgs = [{"role": "system", "content": TL.SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": TL.render_prompt([q], orders)},
                                                 image_part(os.path.join(image_root, r["image"]))]}]
            t0 = time.perf_counter()
            data = await client.chat(msgs, schema=TL.SCHEMA, retries=2)
            dt = time.perf_counter() - t0
            if data and isinstance(TL.validate(message_json(data), [q], orders, allow_onehot=True), dict):
                times.append(dt)
                ok += 1
        await client.close()
        return times, ok, client.stats["usd"]

    times, ok, usd = asyncio.run(go())
    return {"calls": len(picks), "ok": ok, "p50_ms": 1000 * pctl(times, 50) if times else None,
            "p95_ms": 1000 * pctl(times, 95) if times else None, "usd": usd}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--per-task", type=int, default=300, help="300 = the published eval subset (4,872 rows)")
    ap.add_argument("--final", help="Laya-V checkpoint dir (runs Laya-V on the local GPU)")
    ap.add_argument("--laya-rows", help="laya_v_rows.json from an earlier run, instead of --final")
    ap.add_argument("--audit-items")
    ap.add_argument("--audit-verdicts")
    ap.add_argument("--max-usd", type=float, default=30.0, help="spend cap per model")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--latency-calls", type=int, default=50)
    ap.add_argument("--skip", nargs="*", default=[], help="latency, audit, laya")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    report_path = os.path.join(args.out, "bench_vlms.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) else {"systems": {}}

    groups, _ = select_groups(vdata.read_items(args.items, "eval"), args.per_task)
    for g in groups:
        for i, r in enumerate(g):
            r["qkey"] = "%s|%d" % (r["gid"], i)
    print("eval groups", len(groups), "rows", sum(len(g) for g in groups), flush=True)

    audit_groups = []
    if args.audit_items and args.audit_verdicts and "audit" not in args.skip:
        items = vdata.read_items([args.audit_items])
        verdicts = {v["i"]: v.get("human") for v in json.load(open(args.audit_verdicts, encoding="utf-8"))}
        by_img = defaultdict(list)  # one group per image, so the tower sees each image once
        for i, it in enumerate(items):
            if verdicts.get(i) is not None:
                k = len(it["target"])
                by_img[it["image"]].append(dict(it, gid="audit-" + it["gid"], qkey="audit-%d" % i, task="audit",
                                                source="audit", target=[float(j == verdicts[i]) for j in range(k)]))
        audit_groups = list(by_img.values())
        print("audit items", sum(len(g) for g in audit_groups), "images", len(audit_groups), flush=True)

    # ---- Laya-V on the local GPU (or rows from an earlier run). Rows are saved after each part, so a
    # crash in the audit pass keeps the eval rows; they are reused while --per-task is unchanged.
    laya_path = args.laya_rows or os.path.join(args.out, "laya_v_rows.json")
    laya = json.load(open(laya_path)) if os.path.exists(laya_path) else None
    if laya and laya.get("per_task") != args.per_task:
        laya = None
    if args.final and "laya" not in args.skip:
        from eval_vision import laya_v_rows
        laya = laya or {"per_task": args.per_task}
        if "eval" not in laya:
            laya["eval"] = laya_v_rows(args.final, groups, args)
            json.dump(laya, open(laya_path, "w"))
        if "audit" not in laya:
            # audit groups hold 1-2 questions per image, so a 128-row batch would be ~100 large images
            small = argparse.Namespace(**dict(vars(args), rows=min(args.rows, 16)))
            laya["audit"] = laya_v_rows(args.final, audit_groups, small) if audit_groups else []
            json.dump(laya, open(laya_path, "w"))
    if laya:
        report["systems"]["laya_v"] = summarize(laya["eval"], laya["audit"])
        print("laya_v macro_gold %.4f" % report["systems"]["laya_v"]["macro_gold"], flush=True)

    # ---- VLMs
    for model in args.models:
        st, ast = {}, {}
        rows = vlm_rows(groups, args.image_root, model=model, include_teacher_sources=True,
                        cache_path=os.path.join(args.out, "cache", safe(model) + ".jsonl"),
                        concurrency=args.concurrency, max_usd=args.max_usd, stats=st)
        audit_rows = vlm_rows(audit_groups, args.image_root, model=model, include_teacher_sources=True,
                              cache_path=os.path.join(args.out, "cache", safe(model) + ".audit.jsonl"),
                              concurrency=args.concurrency, max_usd=args.max_usd, stats=ast) if audit_groups else []
        s = summarize(rows, audit_rows, laya["eval"] if laya else None)
        cache_file = os.path.join(args.out, "cache", safe(model) + ".jsonl")
        cache_usd = sum(float((json.loads(line).get("usage") or {}).get("cost") or 0)
                        for line in open(cache_file, encoding="utf-8")) if os.path.exists(cache_file) else 0.0
        s["cost"] = {"usd_eval": cache_usd, "usd_per_1k_decisions": 1000 * cache_usd / max(1, len(rows) - s["failed_rows"]),
                     "calls": st["calls"], "schema_fallback": st["schema_fallback"], "http_err": st["http_err"]}
        prev = report["systems"].get(model, {})
        if "latency" not in args.skip:
            s["latency"] = api_latency(model, groups, args.image_root, n=args.latency_calls)
        elif "latency" in prev:
            s["latency"] = prev["latency"]
        s["model"] = model
        report["systems"][model] = s
        print("%s macro_gold %.4f failed %d/%d usd %.2f" % (model, s["macro_gold"], s["failed_rows"], s["n_rows"],
                                                            cache_usd), flush=True)
        report["meta"] = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "per_task": args.per_task,
                          "rows": sum(len(g) for g in groups), "audit_items": len(audit_groups),
                          "prompt": "teacher_label.SYSTEM/render_prompt, temperature 0, reasoning effort low",
                          "families": list(FAMILIES)}
        json.dump(report, open(report_path, "w"), indent=1)
    json.dump(report, open(report_path, "w"), indent=1)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
