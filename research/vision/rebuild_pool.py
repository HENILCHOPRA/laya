"""Rebuild the teacher-labelled image pool on the training box from each row's provenance.

The laptop's upload link is too slow to ship the 2.7 GB pool, and every non-synthetic image comes
from a dataset the box downloads anyway. This recreates $DATA/pool/<row path> with the same
resize the teacher saw (fetch_label_pool.save_jpeg). Synthetic memes come from the Hub upload.

    python research/vision/rebuild_pool.py --data /workspace/data --raw /workspace/raw
"""
import argparse
import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_label_pool import DEFAULT_MAX_SIDE, MAX_SIDE, iter_parquet, save_jpeg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    pool_dir = os.path.join(args.data, "pool")
    rows = [json.loads(l) for l in open(os.path.join(pool_dir, "pool.jsonl"), encoding="utf-8")]
    todo = [r for r in rows if r["domain"] not in ("meme", "meme_base")
            and not os.path.exists(os.path.join(pool_dir, r["path"]))]
    print("to rebuild", len(todo), "of", len(rows), flush=True)

    def out(r):
        return os.path.join(pool_dir, r["path"])

    def side(r):
        return MAX_SIDE.get(r["domain"], DEFAULT_MAX_SIDE)

    by_src = defaultdict(list)
    for r in todo:
        by_src[r["provenance"]["dataset"]].append(r)
    done = 0

    # COCO: straight from train2017
    def coco(r):
        p = os.path.join(args.data, "coco", "train2017", "%012d.jpg" % r["provenance"]["image_id"])
        if os.path.exists(p):  # the HF mirror omits the ~1k images without object annotations
            save_jpeg(Image.open(p), out(r), side(r))
    with ThreadPoolExecutor(32) as ex:
        list(ex.map(coco, by_src.get("coco2017", [])))
    done += len(by_src.get("coco2017", []))

    # pixmo-docs: read only the shards the pool came from
    need = defaultdict(dict)
    for r in by_src.get("allenai/pixmo-docs", []):
        need[r["provenance"]["shard"]][r["provenance"]["image_id"]] = r
    for shard, want in need.items():
        path = os.path.join(args.raw, "pixmo-docs", shard)
        for row in iter_parquet(path):
            r = want.get(row["image_id"])
            if r:
                save_jpeg(Image.open(io.BytesIO(row["image"]["bytes"])), out(r), side(r))
                done += 1

    # CORD: by file + row index
    need = defaultdict(dict)
    for r in by_src.get("naver-clova-ix/cord-v2", []):
        need[r["provenance"]["file"]][r["provenance"]["row"]] = r
    for f, want in need.items():
        for i, row in enumerate(iter_parquet(os.path.join(args.raw, "cord", f))):
            if i in want:
                save_jpeg(Image.open(io.BytesIO(row["image"]["bytes"])), out(want[i]), side(want[i]))
                done += 1

    # Aria-UI desktop: zip members
    rs = by_src.get("Aria-UI/Aria-UI_Data", [])
    if rs:
        z = zipfile.ZipFile(os.path.join(args.raw, "aria", "desktop", "screenshots.zip"))
        for r in rs:
            save_jpeg(Image.open(io.BytesIO(z.read(r["provenance"]["member"]))), out(r), side(r))
            done += 1
    missing = [r["img_id"] for r in rows if r["domain"] != "meme_base" and not os.path.exists(out(r))]
    print("rebuilt", done, "| still missing", len(missing), missing[:5], flush=True)


if __name__ == "__main__":
    main()
