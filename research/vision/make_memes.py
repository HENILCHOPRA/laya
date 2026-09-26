"""Synthetic memes and social posts: teacher-written text rendered onto COCO photos.

For each `meme_base` pool row the teacher writes short post text in one of several styles, from
the photo's COCO captions (related) or from another photo's captions (unrelated, ~30%). The text
is rendered in one of three layouts (classic top/bottom meme, social post card, quote card).
The recorded style / relatedness / language are known ground truth for later label checks.

Appends rows with domain "meme" to <pool>/pool.meme.jsonl and rewrites pool.jsonl.

    python research/vision/make_memes.py --pool E:/LAYA-V/data/pool
"""
import argparse
import asyncio
import json
import os
import random
import sys
import textwrap

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openrouter import Client, message_json  # noqa: E402

STYLES = {
    "relatable": "a relatable everyday joke",
    "sarcastic": "a sarcastic or ironic comment",
    "wholesome": "a wholesome, positive caption",
    "absurd": "an absurd, surreal joke",
    "promotional": "an advertisement for a made-up product or shop, with a call to action",
    "opinion": "a strongly worded opinion on a generic public issue (no real people, parties or countries)",
    "motivational": "a motivational quote",
    "rude": "a mildly rude, mocking comment (no slurs, no real people)",
    "misleading": "a confident but false or misleading factual claim (harmless topic)",
    "wordplay": "a pun or wordplay",
}
LANG_NAMES = {"en": "English", "es": "Spanish", "hi": "Hindi", "pt": "Portuguese", "fr": "French",
              "de": "German", "zh": "Simplified Chinese", "ja": "Japanese"}
LANG_WEIGHTS = {"en": 0.75, "es": 0.05, "hi": 0.04, "pt": 0.04, "fr": 0.03, "de": 0.03, "zh": 0.03, "ja": 0.03}
FONTS_DIR = "C:/Windows/Fonts" if os.name == "nt" else "/usr/share/fonts"
FONT_FOR = {"hi": "Nirmala.ttc", "zh": "msyh.ttc", "ja": "YuGothB.ttc"}

SCHEMA = {"type": "object",
          "properties": {"top": {"type": "string"}, "bottom": {"type": "string"}},
          "required": ["top", "bottom"], "additionalProperties": False}


def font(lang, layout, size):
    name = FONT_FOR.get(lang) or ("impact.ttf" if layout == "classic" else "arialbd.ttf")
    try:
        return ImageFont.truetype(os.path.join(FONTS_DIR, name), size)
    except OSError:
        return ImageFont.load_default(size)


def wrap(text, lang, width_chars):
    if lang in ("zh", "ja"):
        return [text[i:i + max(4, width_chars // 2)] for i in range(0, len(text), max(4, width_chars // 2))]
    return textwrap.wrap(text, width=width_chars) or [""]


def draw_block(draw, lines, fnt, W, y, fill, stroke=0, stroke_fill="black", align_center=True):
    for line in lines:
        w = draw.textlength(line, font=fnt)
        x = (W - w) / 2 if align_center else W * 0.05
        draw.text((x, y), line, font=fnt, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
        y += int(fnt.size * 1.15)
    return y


def render(img, top, bottom, lang, layout):
    img = img.convert("RGB")
    W, H = img.size
    if layout == "classic":
        size = max(18, W // 12)
        fnt = font(lang, layout, size)
        d = ImageDraw.Draw(img)
        chars = max(10, int(W / (size * 0.55)))
        tl = wrap(top.upper() if lang not in ("zh", "ja", "hi") else top, lang, chars) if top else []
        bl = wrap(bottom.upper() if lang not in ("zh", "ja", "hi") else bottom, lang, chars) if bottom else []
        draw_block(d, tl, fnt, W, int(H * 0.03), "white", stroke=max(2, size // 12))
        by = H - int(H * 0.04) - len(bl) * int(size * 1.15)
        draw_block(d, bl, fnt, W, by, "white", stroke=max(2, size // 12))
        return img
    if layout == "social":
        text = (top + " " + bottom).strip()
        size = max(16, W // 26)
        fnt = font(lang, layout, size)
        lines = wrap(text, lang, max(20, int(W / (size * 0.5))))
        head = int(size * 1.15) * len(lines) + size * 3
        canvas = Image.new("RGB", (W, H + head), "white")
        d = ImageDraw.Draw(canvas)
        d.ellipse((W * 0.03, size * 0.4, W * 0.03 + size * 1.6, size * 2.0), fill=(120, 150, 200))
        d.text((W * 0.03 + size * 2, size * 0.6), "@user%d" % random.randint(100, 9999), font=font(lang, "social", int(size * 0.8)), fill=(90, 90, 90))
        draw_block(d, lines, fnt, W, int(size * 2.3), "black", align_center=False)
        canvas.paste(img, (0, head))
        return canvas
    # quote card: darkened photo with centred text
    img = ImageEnhance.Brightness(img).enhance(0.45)
    text = (top + " " + bottom).strip()
    size = max(18, W // 16)
    fnt = font(lang, "quote", size)
    lines = wrap(text, lang, max(12, int(W / (size * 0.55))))
    d = ImageDraw.Draw(img)
    y = (H - len(lines) * int(size * 1.15)) // 2
    draw_block(d, lines, fnt, W, y, "white")
    return img


async def main_async(args):
    rows = []
    with open(os.path.join(args.pool, "pool.coco.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    bases = [r for r in rows if r["domain"] == "meme_base"]
    all_caps = [c for r in rows for c in r["meta"]["captions"][:1]]
    out_path = os.path.join(args.pool, "pool.meme.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["img_id"] for l in f}
    client = Client(concurrency=args.concurrency)
    lock = asyncio.Lock()
    out_f = open(out_path, "a", encoding="utf-8")

    async def one(r):
        iid = "meme-" + r["img_id"]
        if iid in done:
            return
        rng = random.Random(iid)
        style = rng.choice(sorted(STYLES))
        lang = rng.choices(list(LANG_WEIGHTS), weights=list(LANG_WEIGHTS.values()))[0]
        related = rng.random() >= 0.3
        caps = r["meta"]["captions"] if related else [rng.choice(all_caps)]
        layout = rng.choices(["classic", "social", "quote"], weights=[0.6, 0.25, 0.15])[0]
        prompt = ("Write the text for %s about a photo described as: %s\n"
                  "Write it in %s. Keep it short: 'top' at most 12 words, 'bottom' at most 12 words "
                  "('bottom' may be empty). Return JSON {\"top\": ..., \"bottom\": ...}."
                  % (STYLES[style], " / ".join(caps[:3]), LANG_NAMES[lang]))
        data = await client.chat([{"role": "user", "content": prompt}], schema=SCHEMA,
                                 reasoning_effort="low", temperature=0.9)
        parsed = message_json(data) if data else None
        if not parsed or not (parsed.get("top") or parsed.get("bottom")):
            return
        img = Image.open(os.path.join(args.pool, r["path"]))
        out_img = render(img, parsed.get("top", "").strip(), parsed.get("bottom", "").strip(), lang, layout)
        path = os.path.join(args.pool, "images", "meme", iid + ".jpg")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out_img.save(path, "JPEG", quality=90)
        rec = {"img_id": iid, "domain": "meme", "source": "synthetic:coco2017+teacher-text",
               "path": os.path.relpath(path, args.pool).replace("\\", "/"),
               "provenance": {"dataset": "coco2017", "split": "train", "image_id": r["provenance"]["image_id"]},
               "meta": {"text": parsed, "style": style, "lang": lang, "related": related, "layout": layout,
                        "captions": r["meta"]["captions"], "objects": r["meta"]["objects"]}}
        async with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()

    await asyncio.gather(*(one(r) for r in bases))
    out_f.close()
    await client.close()
    # rebuild the merged pool
    with open(os.path.join(args.pool, "pool.jsonl"), "w", encoding="utf-8") as f:
        for src in ["coco", "pixmo", "cord", "aria", "meme"]:
            p = os.path.join(args.pool, "pool.%s.jsonl" % src)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as g:
                    f.write(g.read())
    print("memes:", sum(1 for _ in open(out_path, encoding="utf-8")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--concurrency", type=int, default=64)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
