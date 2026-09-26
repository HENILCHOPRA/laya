"""Write only the eval images the benchmark needs, at the paths the Hub items use (for Colab / small boxes).

build_public.py writes every image of every split; this writes just A-OKVQA validation, PixMo-Docs
validation, CORD validation+test and ScreenSpot, with the same path rules and the same save_img. COCO
val2017 (VQAv2) is a plain download and the teacher pool comes from the Hub, so neither is handled here.

    python research/vision/extract_eval_images.py --data /content/data --raw /content/raw
"""
import argparse
import glob
import io
import os
import re
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_public import iter_parquet, save_img  # noqa: E402


def put(data, path, blob):
    dst = os.path.join(data, path)
    if not os.path.exists(dst):
        save_img(Image.open(io.BytesIO(blob)), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    d, raw = args.data, args.raw
    n = 0
    for f in sorted(glob.glob(os.path.join(raw, "aokvqa", "**", "validation-*.parquet"), recursive=True)):
        for r in iter_parquet(f):
            put(d, "aokvqa/validation/%s.jpg" % r["question_id"], r["image"]["bytes"])
            n += 1
    print("aokvqa", n, flush=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(raw, "pixmo-docs", "*", "validation-*.parquet"))):
        subset = os.path.basename(os.path.dirname(f))
        for r in iter_parquet(f):
            iid = re.sub(r"[^A-Za-z0-9_.-]", "_", r["image_id"])
            put(d, "pixmo/%s/val/%s.jpg" % (subset, iid), r["image"]["bytes"])
            n += 1
    print("pixmo-docs", n, flush=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(raw, "cord", "data", "*.parquet"))):
        if os.path.basename(f).split("-")[0] == "train":
            continue
        stem = os.path.basename(f).rsplit(".", 1)[0]
        for i, r in enumerate(iter_parquet(f)):
            put(d, "cord/%s/%04d.jpg" % (stem, i), r["image"]["bytes"])
            n += 1
    print("cord", n, flush=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(raw, "screenspot", "**", "*.parquet"), recursive=True)):
        for r in iter_parquet(f):
            put(d, "screenspot/%s.jpg" % r["file_name"].rsplit(".", 1)[0], r["image"]["bytes"])
            n += 1
    print("screenspot", n, flush=True)


if __name__ == "__main__":
    main()
