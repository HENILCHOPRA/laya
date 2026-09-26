"""Teacher labels -> Laya-V items (vdata format), with the order check applied.

For images labelled twice (options in label order and shuffled), per question: average the two
distributions when their argmax agrees, drop the question when it does not. Splits by image:
3% eval, 3% calib, rest train. Also writes the audit sample (50 eval images x 6 domains, one
question each) for hand checking.

    python research/vision/build_teacher_items.py --labels E:/LAYA-V/data/labels/teacher.jsonl \
        --pool E:/LAYA-V/data/pool --out E:/LAYA-V/data/items
"""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict

NOTES = [{"note": "Please review this."}, {"channel": "email attachment"}, {"source": "user upload"},
         {"note": "Forwarded for a decision."}, {"ticket": "#4821", "note": "see attached"}]
PROB_FLOOR = 0.01


def h(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def srckey_of(prow):
    """Canonical source-image key, shared with build_public.py items, for hold-out filtering."""
    pv = prow.get("provenance", {})
    ds = pv.get("dataset")
    if ds == "coco2017":
        return "coco:%d" % pv["image_id"]
    if ds == "allenai/pixmo-docs":
        return "pixmo:%s" % pv["image_id"]
    if ds == "naver-clova-ix/cord-v2":
        return "cord:%s:%d" % (pv["file"], pv["row"])
    if ds == "Aria-UI/Aria-UI_Data":
        return "aria:%s" % os.path.basename(pv["member"])
    return None


def split_of(img_id):
    v = h("split:" + img_id) % 100
    return "eval" if v < 3 else "calib" if v < 6 else "train"


def renorm(p):
    p = [max(PROB_FLOOR, x) for x in p]
    s = sum(p)
    return [x / s for x in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-per-domain", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pool = {}
    with open(os.path.join(args.pool, "pool.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pool[r["img_id"]] = r
    recs = defaultdict(dict)
    with open(args.labels, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["img_id"]][r["variant"]] = r

    stats = Counter()
    items, audit_pool = [], defaultdict(list)
    for img_id, variants in recs.items():
        main_r = variants.get("main")
        if main_r is None or img_id not in pool:
            stats["no_main_or_pool"] += 1
            continue
        shuf = {q["qid"]: q for q in variants["shuffle"]["questions"]} if "shuffle" in variants else {}
        prow = pool[img_id]
        split = split_of(img_id)
        rng = random.Random(img_id)
        state = rng.choice(NOTES) if rng.random() < 0.15 else ""
        for q in main_r["questions"]:
            p = q["probs"]
            if q["qid"] in shuf:
                p2 = shuf[q["qid"]]["probs"]
                stats["order_checked"] += 1
                if max(range(len(p)), key=p.__getitem__) != max(range(len(p2)), key=p2.__getitem__):
                    stats["order_disagree_dropped"] += 1
                    continue
                p = renorm([(a + b) / 2 for a, b in zip(p, p2)])
            qdef = {"type": q["type"], "instructions": q["instructions"]}
            if "criteria" in q:
                qdef["criteria"] = q["criteria"]
            it = {"gid": img_id, "image": "pool/" + prow["path"], "domain": prow["domain"], "state": state, "q": qdef,
                  "target": p, "source": main_r["source"], "split": split,
                  "task": "teacher:%s" % prow["domain"], "tkey": q["tkey"], "lang": q["lang"],
                  "srckey": srckey_of(prow)}
            items.append(it)
            stats["items_" + split] += 1
            if split == "eval":
                audit_pool[prow["domain"]].append(it)

    path = os.path.join(args.out, "teacher_items.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # audit sample: one question per image, stratified by domain
    audit = []
    for dom, its in sorted(audit_pool.items()):
        by_img = defaultdict(list)
        for it in its:
            by_img[it["gid"]].append(it)
        imgs = sorted(by_img)
        random.Random("audit-" + dom).shuffle(imgs)
        for g in imgs[: args.audit_per_domain]:
            audit.append(random.Random(g).choice(by_img[g]))
    with open(os.path.join(args.out, "audit_items.jsonl"), "w", encoding="utf-8") as f:
        for it in audit:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    stats["audit"] = len(audit)
    agree = 1 - stats["order_disagree_dropped"] / max(1, stats["order_checked"])
    print(dict(stats), "order-check agreement %.3f" % agree)
    print("by domain:", Counter(it["domain"] for it in items))


if __name__ == "__main__":
    main()
