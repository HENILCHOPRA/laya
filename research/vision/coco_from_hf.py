"""COCO 2017 images + caption annotations from Hugging Face mirrors (images.cocodataset.org is
throttled on some hosts). Writes $DATA/coco/{train,val}2017/%012d.jpg with the original JPEG bytes
and $DATA/coco/annotations/captions_{train,val}2017.json.

    python research/vision/coco_from_hf.py --data /workspace/data --raw /workspace/raw
"""
import argparse
import glob
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    src = os.path.join(args.raw, "coco-hf")
    subprocess.run(["hf", "download", "detection-datasets/coco", "--repo-type", "dataset", "--local-dir", src,
                    "--include", "data/*", "--max-workers", "32"], check=True)
    ann = os.path.join(args.data, "coco")
    subprocess.run(["hf", "download", "merve/coco", "--repo-type", "dataset", "--local-dir", ann,
                    "--include", "annotations/captions_*.json"], check=True)

    import pyarrow.parquet as pq

    def one(path):
        split = "train2017" if "/train-" in path.replace("\\", "/") else "val2017"
        out = os.path.join(args.data, "coco", split)
        os.makedirs(out, exist_ok=True)
        n = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["image_id", "image"]):
            for iid, img in zip(batch.column("image_id").to_pylist(), batch.column("image").to_pylist()):
                p = os.path.join(out, "%012d.jpg" % int(iid))
                if not os.path.exists(p):
                    with open(p, "wb") as f:
                        f.write(img["bytes"])
                n += 1
        return split, n

    files = sorted(glob.glob(os.path.join(src, "data", "*.parquet")))
    with ThreadPoolExecutor(8) as ex:
        tot = {}
        for split, n in ex.map(one, files):
            tot[split] = tot.get(split, 0) + n
    print("coco images", tot)


if __name__ == "__main__":
    main()
