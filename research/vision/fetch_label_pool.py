"""Build the local image pool the teacher labels (runs on a laptop, no GPU).

Writes `<out>/images/<domain>/<img_id>.jpg` and `<out>/pool.jsonl`, one row per image:
    {"img_id", "domain", "path", "source", "provenance": {...}, "meta": {...}}
`provenance` is enough to find the same image again in the source dataset on the training box.

Sources (all Apache 2.0 / CC-BY / ODC-BY):
    photo      COCO train2017 (annotations CC-BY 4.0), fetched by URL
    meme_base  COCO train2017, captions kept for synthetic memes (make_memes.py)
    chart      allenai/pixmo-docs charts + tables + diagrams (ODC-BY)
    document   allenai/pixmo-docs other (ODC-BY)
    receipt    naver-clova-ix/cord-v2 train (CC-BY 4.0) + receipt/form-like pixmo-docs other
    screenshot Aria-UI desktop screenshots (Apache 2.0), read member-by-member over HTTP

Eval images are never drawn: COCO val2017, pixmo-docs validation, CORD validation/test.

    python research/vision/fetch_label_pool.py --out E:/LAYA-V/data/pool
"""
import argparse
import asyncio
import io
import json
import os
import random
import re
import zipfile
from collections import defaultdict

import httpx
from PIL import Image

HF = "https://huggingface.co/datasets/%s/resolve/main/%s"
COCO_ANN = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG = "http://images.cocodataset.org/train2017/%012d.jpg"
ARIA_DESKTOP_ZIP = HF % ("Aria-UI/Aria-UI_Data", "desktop/screenshots.zip")

COUNTS = {"photo": 5000, "meme_base": 2500, "chart": 4000, "document": 3000,
          "receipt_pixmo": 700, "screenshot": 4000}
PIXMO_SHARDS = {  # subset -> shard indices to download (about 2.3k images per shard)
    "charts": [0, 1], "tables": [0], "diagrams": [0], "other": [0, 1],
}
RECEIPT_RE = re.compile(r"receipt|invoice|bill\b|form\b|total amount|subtotal|purchase order", re.I)
MAX_SIDE = {"photo": 1024, "meme_base": 1024}
DEFAULT_MAX_SIDE = 1600
SEED = 20260925


def save_jpeg(img: Image.Image, path: str, max_side: int):
    img = img.convert("RGB")
    s = max(img.size)
    if s > max_side:
        r = max_side / s
        img = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))), Image.BICUBIC)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, "JPEG", quality=90)
    return img.size


def download(url: str, dest: str):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------- COCO
def coco_rows(raw_dir: str, rng: random.Random):
    zpath = download(COCO_ANN, os.path.join(raw_dir, "annotations_trainval2017.zip"))
    with zipfile.ZipFile(zpath) as z:
        caps = json.loads(z.read("annotations/captions_train2017.json"))
        inst = json.loads(z.read("annotations/instances_train2017.json"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    objs = defaultdict(set)
    for a in inst["annotations"]:
        objs[a["image_id"]].add(cats[a["category_id"]])
    del inst
    captions = defaultdict(list)
    for a in caps["annotations"]:
        captions[a["image_id"]].append(a["caption"].strip())
    ids = sorted(captions)
    rng.shuffle(ids)
    need = COUNTS["photo"] + COUNTS["meme_base"]
    out = []
    for i, iid in enumerate(ids[:need]):
        domain = "photo" if i < COUNTS["photo"] else "meme_base"
        out.append({"img_id": "coco-%d" % iid, "domain": domain, "source": "public:coco2017",
                    "provenance": {"dataset": "coco2017", "split": "train", "image_id": iid},
                    "meta": {"objects": sorted(objs.get(iid, [])), "captions": captions[iid]},
                    "_url": COCO_IMG % iid})
    return out


async def fetch_urls(rows, out_dir, conc=32):
    sem = asyncio.Semaphore(conc)
    done = []

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        async def one(row):
            path = os.path.join(out_dir, "images", row["domain"], row["img_id"] + ".jpg")
            if not os.path.exists(path):
                async with sem:
                    for attempt in range(3):
                        try:
                            r = await client.get(row["_url"])
                            r.raise_for_status()
                            size = save_jpeg(Image.open(io.BytesIO(r.content)), path,
                                             MAX_SIDE.get(row["domain"], DEFAULT_MAX_SIDE))
                            row["meta"]["size"] = size
                            break
                        except Exception as e:  # noqa: BLE001
                            if attempt == 2:
                                print("skip", row["img_id"], e)
                                return
                            await asyncio.sleep(2 ** attempt)
            row["path"] = os.path.relpath(path, out_dir).replace("\\", "/")
            row.pop("_url", None)
            done.append(row)

        await asyncio.gather(*(one(r) for r in rows))
    return done


# ---------------------------------------------------------------- parquet sources
def iter_parquet(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            yield row


def pixmo_rows(raw_dir, out_dir, rng):
    rows ={"chart": [], "document": [], "receipt": []}
    for subset, shards in PIXMO_SHARDS.items():
        listing = httpx.get("https://huggingface.co/api/datasets/allenai/pixmo-docs/tree/main/" + subset).json()
        names = sorted(f["path"] for f in listing if f["path"].endswith(".parquet") and "/train-" in f["path"])
        assert names, subset
        for si in shards:
            local = download(HF % ("allenai/pixmo-docs", names[si]), os.path.join(raw_dir, names[si]))
            for row in iter_parquet(local):
                qs = row["questions"]
                if isinstance(qs, dict):
                    qa = [{"q": q, "a": a} for q, a in zip(qs["question"], qs["answer"])]
                else:
                    qa = [{"q": x["question"], "a": x["answer"]} for x in qs]
                text = " ".join(x["q"] for x in qa)
                if subset == "other":
                    domain = "receipt" if RECEIPT_RE.search(text) else "document"
                else:
                    domain = "chart"
                rows[domain].append((subset, names[si], row, qa))
    out = []
    quota = {"chart": COUNTS["chart"], "document": COUNTS["document"], "receipt": COUNTS["receipt_pixmo"]}
    for domain, items in rows.items():
        rng.shuffle(items)
        for subset, shard, row, qa in items[:quota[domain]]:
            iid = "pixmo-%s-%s" % (subset, row["image_id"])
            path = os.path.join(out_dir, "images", domain, iid + ".jpg")
            size = save_jpeg(Image.open(io.BytesIO(row["image"]["bytes"])), path, DEFAULT_MAX_SIDE)
            out.append({"img_id": iid, "domain": domain, "source": "public:pixmo-docs",
                        "path": os.path.relpath(path, out_dir).replace("\\", "/"),
                        "provenance": {"dataset": "allenai/pixmo-docs", "subset": subset, "shard": shard,
                                       "image_id": row["image_id"]},
                        "meta": {"qa": qa, "size": size}})
        print("pixmo", domain, min(len(items), quota[domain]), "of", len(items))
    return out


def cord_rows(raw_dir, out_dir):
    listing = httpx.get("https://huggingface.co/api/datasets/naver-clova-ix/cord-v2/tree/main/data").json()
    names = sorted(f["path"] for f in listing if "/train-" in f["path"])
    out = []
    for name in names:
        local = download(HF % ("naver-clova-ix/cord-v2", name), os.path.join(raw_dir, name))
        for i, row in enumerate(iter_parquet(local)):
            gt = json.loads(row["ground_truth"])
            iid = "cord-%s-%d" % (os.path.basename(name).split("-")[1], i)
            path = os.path.join(out_dir, "images", "receipt", iid + ".jpg")
            size = save_jpeg(Image.open(io.BytesIO(row["image"]["bytes"])), path, DEFAULT_MAX_SIDE)
            out.append({"img_id": iid, "domain": "receipt", "source": "public:cord-v2",
                        "path": os.path.relpath(path, out_dir).replace("\\", "/"),
                        "provenance": {"dataset": "naver-clova-ix/cord-v2", "split": "train", "file": name, "row": i},
                        "meta": {"gt_parse": gt.get("gt_parse"), "size": size}})
    print("cord", len(out))
    return out


# ---------------------------------------------------------------- Aria-UI desktop screenshots
def _signed(url):
    """The Hub redirects to a signed CDN URL that serves range requests; it expires, so re-resolve."""
    r = httpx.head(url, follow_redirects=False, timeout=60)
    return r.headers.get("location", url)


async def _read_member(client, url_box, m):
    """Read one zip member with a single HTTP range request (local header + data)."""
    import struct
    import zlib
    start = m.header_offset
    end = start + 30 + len(m.filename.encode()) + 1024 + m.compress_size
    for attempt in range(4):
        try:
            r = await client.get(url_box[0], headers={"Range": "bytes=%d-%d" % (start, end)})
            if r.status_code in (403, 410):
                url_box[0] = _signed(ARIA_DESKTOP_ZIP)
                continue
            r.raise_for_status()
            b = r.content
            n, e = struct.unpack("<HH", b[26:30])
            data = b[30 + n + e: 30 + n + e + m.compress_size]
            if m.compress_type == 8:
                data = zlib.decompressobj(-15).decompress(data)
            return data
        except Exception:  # noqa: BLE001
            await asyncio.sleep(2 ** attempt)
    return None


async def _aria_fetch(members, out_dir, conc=24):
    url_box = [_signed(ARIA_DESKTOP_ZIP)]
    sem = asyncio.Semaphore(conc)
    out = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        async def one(m):
            iid = "aria-desktop-" + os.path.basename(m.filename).rsplit(".", 1)[0]
            path = os.path.join(out_dir, "images", "screenshot", iid + ".jpg")
            if os.path.exists(path):
                size = Image.open(path).size
            else:
                async with sem:
                    data = await _read_member(client, url_box, m)
                if data is None:
                    print("skip", m.filename)
                    return
                try:
                    size = save_jpeg(Image.open(io.BytesIO(data)), path, DEFAULT_MAX_SIDE)
                except Exception as e:  # noqa: BLE001
                    print("skip", m.filename, e)
                    return
            out.append({"img_id": iid, "domain": "screenshot", "source": "public:aria-ui",
                        "path": os.path.relpath(path, out_dir).replace("\\", "/"),
                        "provenance": {"dataset": "Aria-UI/Aria-UI_Data", "file": "desktop/screenshots.zip",
                                       "member": m.filename},
                        "meta": {"size": size}})
            if len(out) % 250 == 0:
                print("aria", len(out), flush=True)

        await asyncio.gather(*(one(m) for m in members))
    return out


def aria_rows(out_dir, rng):
    from remotezip import RemoteZip
    with RemoteZip(_signed(ARIA_DESKTOP_ZIP)) as z:
        members = [m for m in z.infolist()
                   if m.filename.lower().endswith((".png", ".jpg", ".jpeg")) and "__MACOSX" not in m.filename]
    print("aria desktop members", len(members))
    rng.shuffle(members)
    return asyncio.run(_aria_fetch(members[:COUNTS["screenshot"]], out_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=["coco", "pixmo", "cord", "aria"])
    args = ap.parse_args()
    raw = os.path.join(args.out, "raw")
    os.makedirs(raw, exist_ok=True)
    for src in args.only:
        part = os.path.join(args.out, "pool.%s.jsonl" % src)
        if os.path.exists(part):
            print("done already:", part)
            continue
        rng = random.Random("%d-%s" % (SEED, src))
        if src == "coco":
            rows = asyncio.run(fetch_urls(coco_rows(raw, rng), args.out))
        elif src == "pixmo":
            rows = pixmo_rows(raw, args.out, rng)
        elif src == "cord":
            rows = cord_rows(raw, args.out)
        elif src == "aria":
            rows = aria_rows(args.out, rng)
        else:
            raise SystemExit("unknown source " + src)
        with open(part + ".tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(part + ".tmp", part)
        print(src, "rows:", len(rows))
    # merge
    with open(os.path.join(args.out, "pool.jsonl"), "w", encoding="utf-8") as f:
        for src in ["coco", "pixmo", "cord", "aria", "meme"]:
            p = os.path.join(args.out, "pool.%s.jsonl" % src)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as g:
                    f.write(g.read())


if __name__ == "__main__":
    main()
