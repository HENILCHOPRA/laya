"""End-to-end latency of the released Laya-V checkpoint through `laya.Router`, as a user calls it.

Unlike latency_gate.py (random adapter weights, synthetic input), this loads the real checkpoint and
times `Router.predict` on real eval images: 1 image + 1 question and 1 image + 10 questions, at each
patch budget (low 256 / mid 576 / high 1,024). Wall time includes image preprocessing and routing.

    python research/vision/bench_latency.py --ckpt /workspace/ckpt/laya-v --items data/items/stage_b.jsonl \
        --image-root data --usd-per-hour 0.20 --out research/vision/results/latency_t4_real.json
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latency_gate import pctl  # noqa: E402
import vdata  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--usd-per-hour", type=float, help="GPU rental price, for $ per 1,000 decisions")
    args = ap.parse_args()
    os.environ["LAYA_VISION_REPO"] = args.ckpt

    import torch
    import laya
    from PIL import Image

    groups = [g for g in vdata.group_items([r for r in vdata.read_items(args.items, "eval") if r.get("image")])
              if len(g) >= 10]
    rng = random.Random(0)
    rng.shuffle(groups)
    pool = groups[:args.repeats]
    images = [Image.open(os.path.join(args.image_root, g[0]["image"])).convert("RGB") for g in pool]

    def qs(g, n):
        return {"q%d" % i: dict(r["q"]) for i, r in enumerate(g[:n])}

    router = laya.Router(device="cuda")
    router.preload(["vision"])
    res = {"gpu": torch.cuda.get_device_name(0), "ckpt": args.ckpt, "repeats": len(pool)}
    for detail in ("low", "mid", "high"):
        for n in (1, 10):
            for i in range(args.warmup):
                router.predict({"image": images[i % len(images)], "image_detail": detail}, qs(pool[i % len(pool)], n))
            times = []
            for img, g in zip(images, pool):
                t0 = time.perf_counter()
                out = router.predict({"image": img, "image_detail": detail}, qs(g, n))
                times.append(time.perf_counter() - t0)
                assert out["routing"]["model"] == "vision"
            key = "%s_q%d" % (detail, n)
            res[key] = {"p50_ms": 1000 * pctl(times, 50), "p95_ms": 1000 * pctl(times, 95),
                        "decisions_per_s": n / pctl(times, 50)}
            print(key, json.dumps(res[key]), flush=True)
    if args.usd_per_hour:
        dps = res["mid_q10"]["decisions_per_s"]
        res["usd_per_hour"] = args.usd_per_hour
        res["usd_per_1k_decisions"] = 1000 * args.usd_per_hour / 3600 / dps
        res["usd_per_1k_decisions_q1"] = 1000 * args.usd_per_hour / 3600 / res["mid_q1"]["decisions_per_s"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
