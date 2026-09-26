"""Stage A items: caption matching with hard negatives (A100; SigLIP 2 text tower on GPU).

For every image in the caption tables (COCO, pixmo-docs QA-as-caption, PixMo-Cap), one caption
is embedded with the SigLIP 2 text tower. Hard negatives for an image are captions of OTHER
images from the same source whose embeddings rank 5-50 by cosine (the top few are skipped:
near-duplicate captions of different photos are often also true). Two rows per image:
    choice: "Which caption describes this image?"  A-D, one true caption, three hard negatives
    noul:   "Does this caption describe the image? <caption>"  true or hard negative, 50/50

    python research/vision/build_stage_a.py --data /workspace/data --max-train 300000
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch

CHOICE_INS = {
    "en": "Which caption describes this image?", "es": "¿Qué descripción corresponde a esta imagen?",
    "de": "Welche Beschreibung passt zu diesem Bild?", "fr": "Quelle légende décrit cette image ?",
    "hi": "कौन सा कैप्शन इस छवि का वर्णन करता है?", "zh": "哪个描述符合这张图片？", "ja": "この画像を説明しているキャプションはどれですか？",
    "pt": "Qual legenda descreve esta imagem?",
}
NOUL_INS = {
    "en": "Does this caption describe the image? Caption: %s", "es": "¿Esta descripción corresponde a la imagen? Descripción: %s",
    "de": "Beschreibt diese Bildunterschrift das Bild? Bildunterschrift: %s", "fr": "Cette légende décrit-elle l'image ? Légende : %s",
    "hi": "क्या यह कैप्शन छवि का वर्णन करता है? कैप्शन: %s", "zh": "这个描述与图片相符吗？描述：%s",
    "ja": "このキャプションは画像を説明していますか？キャプション：%s", "pt": "Esta legenda descreve a imagem? Legenda: %s",
}
SMOOTH = 0.03


def pick_lang(rng):
    return "en" if rng.random() < 0.8 else rng.choice([k for k in CHOICE_INS if k != "en"])


@torch.no_grad()
def embed(texts, model, tok, device, bs=1024):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding="max_length", truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.get_text_features(**enc)
        if hasattr(f, "pooler_output"):
            f = f.pooler_output
        out.append(torch.nn.functional.normalize(f.float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def hard_negatives(E, lo=5, hi=50, chunk=4096):
    """For each row, indices of rows at similarity ranks [lo, hi) (excluding itself)."""
    res = []
    for i in range(0, E.size(0), chunk):
        s = E[i:i + chunk] @ E.T
        idx = torch.arange(i, min(i + chunk, E.size(0)), device=E.device)
        s[torch.arange(len(idx)), idx] = -2
        res.append(s.topk(hi, dim=-1).indices[:, lo:].cpu())
    return torch.cat(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--encoder", default="google/siglip2-so400m-patch16-naflex")
    ap.add_argument("--max-train", type=int, default=300000)
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer

    rows = []
    for f in sorted(glob.glob(os.path.join(args.data, "items", "captions_*.jsonl"))):
        with open(f, encoding="utf-8") as g:
            rows += [json.loads(l) for l in g]
    rows = [r for r in rows if r.get("captions") and os.path.exists(os.path.join(args.data, r["image"]))]
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import vdata
    held = vdata.holdout_keys(os.path.join(args.data, "items", "teacher_items.jsonl"))
    rows = [r for r in rows if not (r["split"] == "train" and r.get("srckey") in held)]
    rng = random.Random(0)
    train = [r for r in rows if r["split"] == "train"]
    rng.shuffle(train)
    train = train[: args.max_train]
    evals = [r for r in rows if r["split"] != "train"]
    rows = train + evals
    print("images: train %d eval %d" % (len(train), len(evals)), flush=True)

    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder, torch_dtype=torch.bfloat16).to(device).eval()
    cap = [rng.choice(r["captions"]) for r in rows]
    by_src = defaultdict(list)
    for i, r in enumerate(rows):
        by_src[(r["source"], r["split"] != "train")].append(i)
    negs = {}
    for key, idxs in by_src.items():
        E = embed([cap[i] for i in idxs], model, tok, device)
        H = hard_negatives(E, lo=5, hi=min(50, len(idxs) - 1))
        for j, i in enumerate(idxs):
            negs[i] = [idxs[k] for k in H[j].tolist()]
        print("negatives", key, len(idxs), flush=True)
    del model
    torch.cuda.empty_cache()

    # Symmetric option sets: disjoint clusters of 4 mutually similar images share one option set
    # (their 4 captions) and each image in the cluster is asked the same question. Every option is
    # the right answer for exactly one image, so no text-only cue (such as "the caption most
    # similar to the others") beats chance; only the image decides.
    out = open(os.path.join(args.data, "items", "stage_a.jsonl"), "w", encoding="utf-8")
    n = 0
    used = set()
    order_idx = list(range(len(rows)))
    rng.shuffle(order_idx)
    for i in order_idx:
        if i in used:
            continue
        mates = list(dict.fromkeys(k for k in negs.get(i, []) if k not in used and k != i))[:3]
        cluster = [i] + mates
        if len(mates) < 3 or len({cap[k] for k in cluster}) < 4:
            continue
        used.update(cluster)
        opts = [cap[c] for c in cluster]
        for k in cluster:
            r = rows[k]
            split = "train" if r["split"] == "train" else "eval"
            perm = list(range(4))
            rng.shuffle(perm)
            crit = {key: opts[o] for key, o in zip("ABCD", perm)}
            tgt = [SMOOTH / 3] * 4
            tgt[perm.index(cluster.index(k))] = 1 - SMOOTH
            base = {"gid": r["gid"], "image": r["image"], "domain": r["domain"], "state": "", "source": r["source"],
                    "split": split}
            if "srckey" in r:
                base["srckey"] = r["srckey"]
            lang = pick_lang(rng)
            out.write(json.dumps(dict(base, q={"type": "choice", "instructions": CHOICE_INS[lang], "criteria": crit},
                                      target=tgt, task="stageA:choice"), ensure_ascii=False) + "\n")
            truth = rng.random() < 0.5
            shown = cap[k] if truth else cap[rng.choice([c for c in cluster if c != k])]
            lang = pick_lang(rng)
            out.write(json.dumps(dict(base, q={"type": "noul", "instructions": NOUL_INS[lang] % shown},
                                      target=[SMOOTH, 1 - SMOOTH] if truth else [1 - SMOOTH, SMOOTH],
                                      task="stageA:noul"), ensure_ascii=False) + "\n")
            n += 2
    out.close()
    print("stage A rows", n, flush=True)


if __name__ == "__main__":
    main()
