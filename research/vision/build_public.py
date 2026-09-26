"""Public datasets -> Laya-V images + items (runs on the A100 box after a100_setup.sh).

Writes images under $DATA/<source>/... and items (vdata format) under $DATA/items/:
    public_items.jsonl    stage B decisions: VQAv2, A-OKVQA, pixmo-docs QA, CORD, Aria-UI (+ eval splits,
                          ScreenSpot eval)
    captions.jsonl        stage A caption table: {"gid", "image", "domain", "captions": [...], "split"}
    manifest.json         source, license, counts

Eval splits never reach training:
    VQAv2 val -> only questions on COCO val2017 images (train2017 contains the rest of val2014)
    A-OKVQA val, pixmo-docs validation, CORD validation+test, ScreenSpot (all) -> eval

    python research/vision/build_public.py --data /workspace/data --raw /workspace/raw
"""
import argparse
import glob
import io
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

SMOOTH = 0.03
GRID = ["top_left", "top_center", "top_right", "middle_left", "center", "middle_right", "bottom_left",
        "bottom_center", "bottom_right"]
GRID_CRIT = {k: k.replace("_", " ") + " of the screen" for k in GRID}
LICENSES = {"coco2017": "CC-BY-4.0 (annotations); images Flickr terms", "vqav2": "CC-BY-4.0",
            "a-okvqa": "Apache-2.0", "pixmo-docs": "ODC-BY", "pixmo-cap": "ODC-BY-1.0", "cord-v2": "CC-BY-4.0",
            "aria-ui": "Apache-2.0", "screenspot": "Apache-2.0"}


def onehot(k, i, smooth=SMOOTH):
    t = [smooth / max(1, k - 1)] * k
    t[i] = 1 - smooth
    return t


def floor_norm(p, floor=0.01):
    p = [max(floor, x) for x in p]
    s = sum(p)
    return [x / s for x in p]


def save_img(img, path, max_side=1600):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img.convert("RGB")
    s = max(img.size)
    if s > max_side:
        r = max_side / s
        img = img.resize((max(1, round(img.width * r)), max(1, round(img.height * r))), Image.BICUBIC)
    img.save(path, "JPEG", quality=92)


def iter_parquet(path, columns=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        yield from batch.to_pylist()


class Writer:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.counts = Counter()

    def add(self, it):
        self.f.write(json.dumps(it, ensure_ascii=False) + "\n")
        self.counts[(it["source"], it["split"])] += 1

    def close(self):
        self.f.close()


def item(gid, image, domain, q, target, source, split, task, state="", srckey=None):
    it = {"gid": gid, "image": image, "domain": domain, "state": state, "q": q, "target": target,
          "source": source, "split": split, "task": task}
    if srckey:
        it["srckey"] = srckey  # canonical source-image key, for the teacher hold-out filter
    return it


# ============================================================ COCO captions + VQAv2
def coco_and_vqa(data, raw, W, caps, rng, n_train_q=40000):
    ann = os.path.join(data, "coco", "annotations")
    val_ids = set()
    for split, fname in (("train", "captions_train2017.json"), ("val", "captions_val2017.json")):
        d = json.load(open(os.path.join(ann, fname)))
        by = defaultdict(list)
        for a in d["annotations"]:
            by[a["image_id"]].append(a["caption"].strip())
        for iid, cs in by.items():
            if split == "val":
                val_ids.add(iid)
            caps.write(json.dumps({"gid": "coco-%d" % iid, "image": "coco/%s2017/%012d.jpg" % (split, iid),
                                   "domain": "photo", "captions": cs, "source": "public:coco2017",
                                   "srckey": "coco:%d" % iid,
                                   "split": "train" if split == "train" else "eval"}) + "\n")
    vq = os.path.join(raw, "vqa")
    for split in ("Train", "Val"):
        qs = json.load(open(os.path.join(vq, "v2_OpenEnded_mscoco_%s2014_questions.json" % split.lower())))["questions"]
        an = {a["question_id"]: a for a in json.load(open(os.path.join(vq, "v2_mscoco_%s2014_annotations.json"
                                                                       % split.lower())))["annotations"]}
        # answer vocab per question-type for choice distractors
        vocab = defaultdict(Counter)
        for a in an.values():
            if a["answer_type"] == "other":
                vocab[a["question_type"]][a["multiple_choice_answer"]] += 1
        rng.shuffle(qs)
        n = {"yes/no": 0, "number": 0, "other": 0}
        cap = {"yes/no": n_train_q * 0.45, "number": n_train_q * 0.2, "other": n_train_q * 0.35} if split == "Train" \
            else {"yes/no": 4000, "number": 2000, "other": 4000}
        for q in qs:
            iid = q["image_id"]
            if split == "Val" and iid not in val_ids:
                continue  # val2014 images outside val2017 are in train2017 (stage A): never evaluate on them
            a = an[q["question_id"]]
            at = a["answer_type"]
            if n[at] >= cap[at]:
                continue
            answers = [x["answer"].strip().lower() for x in a["answers"]]
            img = "coco/%s2017/%012d.jpg" % ("val" if split == "Val" else "train", iid)
            gid = "coco-%d" % iid
            sp = "eval" if split == "Val" else "train"
            if at == "yes/no":
                f = sum(x == "yes" for x in answers) / len(answers)
                t = floor_norm([1 - f, f])
                qd = {"type": "noul", "instructions": q["question"]}
            elif at == "number":
                lv = []
                for x in answers:
                    if x.isdigit():
                        lv.append(min(4, int(x)))
                if len(lv) < 6:
                    continue
                t = floor_norm([lv.count(i) / len(lv) for i in range(5)])
                qd = {"type": "score", "instructions": q["question"],
                      "criteria": ["none", "one", "two", "three", "four or more"]}
            else:
                top = a["multiple_choice_answer"].strip().lower()
                c = Counter(answers)
                if c[top] < 3 or len(top) > 40:
                    continue
                pool = [w for w, _ in vocab[a["question_type"]].most_common(60) if w not in c and len(w) <= 40]
                if len(pool) < 3:
                    continue
                opts = [top] + rng.sample(pool, 3)
                others = [w for w in c if w != top and len(w) <= 40][:1]
                opts = list(dict.fromkeys(opts[:4 - len(others)] + others))
                rng.shuffle(opts)
                t = floor_norm([c.get(o, 0) / len(answers) for o in opts])
                qd = {"type": "choice", "instructions": q["question"], "criteria": {o: "" for o in opts}}
            n[at] += 1
            W.add(item(gid, img, "photo", qd, t, "public:vqav2", sp, "vqav2:%s" % at, srckey="coco:%d" % iid))
        print("vqa", split, n, flush=True)


# ============================================================ A-OKVQA
def aokvqa(data, raw, W):
    for split, sp in (("train", "train"), ("validation", "eval")):
        for f in sorted(glob.glob(os.path.join(raw, "aokvqa", "**", "%s-*.parquet" % split), recursive=True)):
            for r in iter_parquet(f):
                path = "aokvqa/%s/%s.jpg" % (split, r["question_id"])
                save_img(Image.open(io.BytesIO(r["image"]["bytes"])), os.path.join(data, path))
                choices = list(dict.fromkeys(str(c) for c in r["choices"]))
                if len(choices) != len(r["choices"]):
                    continue
                qd = {"type": "choice", "instructions": r["question"], "criteria": {c: "" for c in choices}}
                W.add(item("aok-" + r["question_id"], path, "photo", qd,
                           onehot(len(choices), int(r["correct_choice_idx"])), "public:a-okvqa", sp, "aokvqa"))


# ============================================================ pixmo-docs
NUM = re.compile(r"^-?\$?\d[\d,]*(\.\d+)?%?$")


def perturb_number(a, rng):
    s = a.replace("$", "").replace(",", "").replace("%", "")
    try:
        v = float(s)
    except ValueError:
        return []
    outs = set()
    for f in (0.5, 0.8, 0.9, 1.1, 1.25, 2.0):
        w = v * f
        w = round(w) if float(v).is_integer() else round(w, max(1, len(s.split(".")[-1]) if "." in s else 1))
        if w != v:
            txt = ("{:,}".format(w) if "," in a else str(w))
            outs.add(("$" if a.startswith("$") else "") + txt + ("%" if a.endswith("%") else ""))
    return list(outs)


def pixmo_docs(data, raw, W, caps, rng):
    for f in sorted(glob.glob(os.path.join(raw, "pixmo-docs", "*", "*.parquet"))):
        subset = os.path.basename(os.path.dirname(f))
        is_val = "validation" in os.path.basename(f)
        sp = "eval" if is_val else "train"
        domain = "document" if subset == "other" else "chart"
        for r in iter_parquet(f):
            qs = r["questions"]
            qa = list(zip(qs["question"], qs["answer"])) if isinstance(qs, dict) else [(x["question"], x["answer"]) for x in qs]
            iid = re.sub(r"[^A-Za-z0-9_.-]", "_", r["image_id"])
            path = "pixmo/%s/%s/%s.jpg" % (subset, "val" if is_val else "train", iid)
            save_img(Image.open(io.BytesIO(r["image"]["bytes"])), os.path.join(data, path))
            gid = "pixmo-%s-%s" % (subset, iid)
            sk = "pixmo:%s" % r["image_id"]
            short = [(q, a.strip()) for q, a in qa if 0 < len(a.strip()) <= 60]
            caps.write(json.dumps({"gid": gid, "image": path, "domain": domain, "source": "public:pixmo-docs", "srckey": sk,
                                   "captions": ["%s %s" % (q.strip(), a) for q, a in short[:5]], "split": sp}) + "\n")
            answers = [a for _, a in short]
            for q, a in short[:3]:
                low = a.lower().rstrip(".")
                if low in ("yes", "no"):
                    qd = {"type": "noul", "instructions": q}
                    W.add(item(gid, path, domain, qd, onehot(2, int(low == "yes")), "public:pixmo-docs", sp,
                               "pixmo:yesno", srckey=sk))
                    continue
                cands = perturb_number(a, rng) if NUM.match(a) else []
                # Distractors of the same kind (numeric or not) and similar length, so the right
                # option is not the only one that even looks like an answer.
                cands += [x for x in answers if x != a and bool(NUM.match(x)) == bool(NUM.match(a))
                          and x.lower().rstrip(".") not in ("yes", "no") and 0.5 <= len(x) / max(1, len(a)) <= 2.0]
                cands = list(dict.fromkeys(c for c in cands if c.lower() != a.lower()))
                if len(cands) >= 2:
                    opts = [a] + rng.sample(cands, min(3, len(cands)))
                    rng.shuffle(opts)
                    qd = {"type": "choice", "instructions": q, "criteria": {o: "" for o in opts}}
                    W.add(item(gid, path, domain, qd, onehot(len(opts), opts.index(a)), "public:pixmo-docs", sp,
                               "pixmo:choice", srckey=sk))
                if cands and rng.random() < 0.5:
                    truth = rng.random() < 0.5
                    shown = a if truth else rng.choice(cands)
                    qd = {"type": "noul", "instructions": "%s Is the answer %s?" % (q.rstrip(), shown)}
                    W.add(item(gid, path, domain, qd, onehot(2, int(truth)), "public:pixmo-docs", sp, "pixmo:verify",
                               srckey=sk))
        print("pixmo-docs", f, flush=True)


# ============================================================ PixMo-Cap (stage A only; images by URL)
def pixmo_cap(data, raw, caps, rng, n=100000, workers=64, deadline_s=5400):
    import time

    import httpx
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "pixmo-cap", "**", "*.parquet"), recursive=True)):
        rows += list(iter_parquet(f, columns=["image_url", "caption"]))
    rng.shuffle(rows)
    rows = rows[: int(n * 1.4)]
    t0 = time.time()
    ok = [0]

    def one(i_r):
        i, r = i_r
        if ok[0] >= n or time.time() - t0 > deadline_s:
            return None
        path = "pixmo-cap/%06d.jpg" % i
        full = os.path.join(data, path)
        if not os.path.exists(full):
            try:
                b = httpx.get(r["image_url"], timeout=15, follow_redirects=True).content
                save_img(Image.open(io.BytesIO(b)), full, 1024)
            except Exception:  # noqa: BLE001 - dead URLs are expected
                return None
        ok[0] += 1
        sents = re.split(r"(?<=[.!?])\s+", r["caption"].strip())
        return {"gid": "pixcap-%06d" % i, "image": path, "domain": "photo", "source": "public:pixmo-cap",
                "captions": [" ".join(sents[:2])], "split": "train"}

    with ThreadPoolExecutor(workers) as ex:
        for rec in ex.map(one, enumerate(rows)):
            if rec:
                caps.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("pixmo-cap images", ok[0], flush=True)


# ============================================================ CORD
def _num(s):
    s = re.sub(r"[^\d.,]", "", str(s or ""))
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def cord(data, raw, W, rng):
    all_items = []
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "cord", "data", "*.parquet"))):
        split = os.path.basename(f).split("-")[0]
        for i, r in enumerate(iter_parquet(f)):
            rows.append((split, i, r, "data/" + os.path.basename(f)))
    for split, i, r, _ in rows:
        gt = json.loads(r["ground_truth"]).get("gt_parse", {})
        menu = gt.get("menu", [])
        menu = menu if isinstance(menu, list) else [menu]
        all_items += [m.get("nm") for m in menu if isinstance(m, dict) and isinstance(m.get("nm"), str)]
    for split, i, r, fname in rows:
        sp = "train" if split == "train" else "eval"
        stem = os.path.basename(fname).rsplit(".", 1)[0]
        path = "cord/%s/%04d.jpg" % (stem, i)
        save_img(Image.open(io.BytesIO(r["image"]["bytes"])), os.path.join(data, path))
        gid = "cord-%s-%d" % (stem, i)
        sk = "cord:%s:%d" % (fname, i)
        gt = json.loads(r["ground_truth"]).get("gt_parse", {})
        menu = gt.get("menu", [])
        menu = [m for m in (menu if isinstance(menu, list) else [menu]) if isinstance(m, dict)]
        names = [m["nm"] for m in menu if isinstance(m.get("nm"), str)]
        total = gt.get("total", {}) if isinstance(gt.get("total"), dict) else {}
        sub = gt.get("sub_total", {}) if isinstance(gt.get("sub_total"), dict) else {}

        def add(q, t, task):
            W.add(item(gid, path, "receipt", q, t, "public:cord-v2", sp, task, srckey=sk))

        if names:
            truth = rng.random() < 0.5
            nm = rng.choice(names) if truth else rng.choice([x for x in all_items if x not in names] or ["Coffee"])
            add({"type": "noul", "instructions": "Does the receipt include %s?" % nm}, onehot(2, int(truth)), "cord:item")
            n = len(menu)
            lvl = 0 if n <= 1 else 1 if n <= 3 else 2 if n <= 7 else 3
            add({"type": "score", "instructions": "How many line items are listed?",
                 "criteria": ["one", "two or three", "four to seven", "eight or more"]}, onehot(4, lvl), "cord:count")
        tp = _num(total.get("total_price"))
        if tp:
            f = rng.choice([0.5, 0.8, 1.25, 2.0])
            thr = int(round(tp * f, -2)) if tp > 1000 else round(tp * f, 2)
            add({"type": "noul", "instructions": "Is the total amount greater than %s?" % "{:,}".format(thr)},
                onehot(2, int(tp > thr)), "cord:total")
        has_tax = any(k in sub for k in ("tax_price", "service_price"))
        add({"type": "noul", "instructions": "Is tax or a service charge shown?"}, onehot(2, int(has_tax)), "cord:tax")
        if any(k in total for k in ("cashprice", "creditcardprice", "emoneyprice")):
            lab = "card" if "creditcardprice" in total else "mobile_wallet" if "emoneyprice" in total else "cash"
            crit = {"cash": "", "card": "credit or debit card", "mobile_wallet": "phone or e-money", "not_shown": ""}
            add({"type": "choice", "instructions": "How was it paid?", "criteria": crit},
                onehot(4, list(crit).index(lab)), "cord:payment")
    print("cord rows", len(rows), flush=True)


# ============================================================ UI grounding (Aria-UI desktop, ScreenSpot)
def grid_cell(cx, cy):
    col = 0 if cx < 1 / 3 else 1 if cx < 2 / 3 else 2
    row = 0 if cy < 1 / 3 else 1 if cy < 2 / 3 else 2
    return row * 3 + col


def grid_question(instr):
    return {"type": "choice", "instructions": "Where on the screen is the element for: \"%s\"?" % instr.strip(),
            "criteria": GRID_CRIT}


def aria(data, raw, W, rng, n_shots=8000, per_shot=4):
    import ijson
    zpath = os.path.join(raw, "aria", "desktop", "screenshots.zip")
    z = zipfile.ZipFile(zpath)
    names = {os.path.basename(n): n for n in z.namelist() if "__MACOSX" not in n and n.lower().endswith(".png")}
    shots = 0
    with open(os.path.join(raw, "aria", "desktop", "aria_ui_desktop_with_instructions.json"), "rb") as f:
        for rec in ijson.items(f, "item"):
            if shots >= n_shots:
                break
            member = names.get(rec.get("img_file"))
            els = [e for e in rec.get("elements", []) if e.get("instructions") and e.get("bbox_norm")]
            if not member or not els:
                continue
            path = "aria/desktop/%s.jpg" % rec["img_file"].rsplit(".", 1)[0]
            try:
                save_img(Image.open(io.BytesIO(z.read(member))), os.path.join(data, path))
            except Exception:  # noqa: BLE001
                continue
            gid = "aria-desktop-" + rec["img_file"].rsplit(".", 1)[0]
            for e in rng.sample(els, min(per_shot, len(els))):
                x1, y1, x2, y2 = [float(v) / 1000 for v in e["bbox_norm"]]
                cell = grid_cell((x1 + x2) / 2, (y1 + y2) / 2)
                W.add(item(gid, path, "screenshot", grid_question(rng.choice(e["instructions"])),
                           onehot(9, cell, 0.06), "public:aria-ui", "train", "aria:grid",
                           srckey="aria:%s" % rec["img_file"]))
            shots += 1
    print("aria screenshots", shots, flush=True)


def screenspot(data, raw, W):
    n = 0
    for f in sorted(glob.glob(os.path.join(raw, "screenspot", "**", "*.parquet"), recursive=True)):
        for i, r in enumerate(iter_parquet(f)):
            path = "screenspot/%s.jpg" % r["file_name"].rsplit(".", 1)[0]
            save_img(Image.open(io.BytesIO(r["image"]["bytes"])), os.path.join(data, path))
            x1, y1, x2, y2 = r["bbox"]
            gid = "sspot-%s" % r["file_name"].rsplit(".", 1)[0]
            W.add(item(gid, path, "screenshot", grid_question(r["instruction"]), onehot(9, grid_cell((x1 + x2) / 2, (y1 + y2) / 2), 0.0),
                       "public:screenspot", "eval", "screenspot:%s" % r.get("data_source", "")))
            n += 1
    print("screenspot rows", n, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--pixcap-n", type=int, default=100000)
    ap.add_argument("--tag", default="", help="suffix for output file names (e.g. a validation-only pass)")
    args = ap.parse_args()
    out = os.path.join(args.data, "items")
    os.makedirs(out, exist_ok=True)
    steps = args.only or ["coco", "aokvqa", "pixmo", "cord", "aria", "screenspot", "pixcap"]
    for s in steps:
        rng = random.Random("public-" + s)
        W = Writer(os.path.join(out, "public_%s%s.jsonl" % (s, args.tag)))
        caps = open(os.path.join(out, "captions_%s%s.jsonl" % (s, args.tag)), "w", encoding="utf-8")
        if s == "coco":
            coco_and_vqa(args.data, args.raw, W, caps, rng)
        elif s == "aokvqa":
            aokvqa(args.data, args.raw, W)
        elif s == "pixmo":
            pixmo_docs(args.data, args.raw, W, caps, rng)
        elif s == "cord":
            cord(args.data, args.raw, W, rng)
        elif s == "aria":
            aria(args.data, args.raw, W, rng)
        elif s == "screenspot":
            screenspot(args.data, args.raw, W)
        elif s == "pixcap":
            pixmo_cap(args.data, args.raw, caps, rng, n=args.pixcap_n)
        W.close()
        caps.close()
        with open(os.path.join(out, "manifest_%s%s.json" % (s, args.tag)), "w") as f:
            json.dump({"counts": {"%s|%s" % k: v for k, v in W.counts.items()}, "licenses": LICENSES}, f, indent=1)
        print("DONE", s, dict(W.counts), flush=True)


if __name__ == "__main__":
    main()
