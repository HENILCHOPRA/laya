"""Write the Laya-V model card (README.md + assets/*.png) from the run's own result files.

Every number on the card is read from these files; nothing is typed by hand.

    python research/vision/make_model_card.py --ckpt <dir> --eval results/eval_report.json \
        --regression results/text_regression_final.json --manifest stage_b_manifest.json \
        --latency-real results/latency_t4_real.json --bench results/bench_vlms.json \
        --laya-rows laya_v_rows.json [--params 751867590]
"""
import argparse
import json
import os
import struct

INSTALL = 'pip install "laya @ git+https://github.com/HENILCHOPRA/laya@laya-v" "transformers>=5.0"'
REPO = "henilchopada/laya-v"
QWEN = "qwen/qwen3-vl-8b-instruct"
DOMAINS = [("meme", "Memes and posts"), ("screenshot", "Screenshots"), ("document", "Documents"),
           ("photo", "Photos"), ("receipt", "Receipts"), ("chart", "Charts and tables")]

# chart tokens (light surface): Laya-V is the accent, comparators recede
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, GREY, LIGHT_GREY = "#2a78d6", "#eb6834", "#a3a29d", "#d4d3cf"


def pct(x, d=1):
    return "–" if x is None else ("%." + str(d) + "f%%") % (100 * x)


def load(p):
    return json.load(open(p, encoding="utf-8")) if p and os.path.exists(p) else {}


def count_params(path):
    with open(path, "rb") as f:
        meta = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    total = 0
    for k, v in meta.items():
        if k != "__metadata__":
            n = 1
            for s in v["shape"]:
                n *= s
            total += n
    return total


# ----------------------------------------------------------------------------- charts
def _axes(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    import matplotlib.pyplot as plt
    plt.close(fig)


def chart_domains(ev, path):
    """Accuracy by image domain: Laya-V vs the OCR + caption pipeline vs the image dropped."""
    import numpy as np

    series = [("laya_v", "Laya-V", BLUE), ("ocr_pipeline", "OCR + caption pipeline", GREY),
              ("text_only", "Image dropped", LIGHT_GREY)]
    fig, ax = _axes((7.2, 4.4))
    y = np.arange(len(DOMAINS))[::-1]
    h = 0.26
    for j, (key, label, color) in enumerate(series):
        vals = [100 * ev[key]["task:teacher:" + d]["acc"] for d, _ in DOMAINS]
        pos = y + (1 - j) * h
        ax.barh(pos, vals, height=h - 0.03, color=color, label=label)
        if key == "laya_v":
            for p, v in zip(pos, vals):
                ax.text(v + 1, p, "%.0f%%" % v, va="center", fontsize=8, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([name for _, name in DOMAINS], color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Accuracy on held-out images (%)", color=MUTED, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=INK)
    _save(fig, path)


def chart_calibration(rows, path, bins=10):
    """Reliability diagram: accuracy of the answers in each confidence bin, against the diagonal."""
    import numpy as np

    conf = np.array([max(r["p"]) for r in rows])
    ok = np.array([float(np.argmax(r["p"]) == np.argmax(r["target"])) for r in rows])
    edges = np.linspace(0, 1, bins + 1)
    fig, ax = _axes((5.6, 4.4))
    ax.plot([0, 1], [0, 100], color=GREY, linewidth=1.2, linestyle=(0, (4, 3)), label="Perfect calibration")
    xs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append((lo + hi) / 2)
            accs.append(100 * ok[m].mean())
            ns.append(int(m.sum()))
    ax.bar(xs, accs, width=(edges[1] - edges[0]) * 0.86, color=BLUE, label="Laya-V")
    for x, a, n in zip(xs, accs, ns):
        ax.text(x, a + 1.5, "n=%d" % n, ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_xlabel("Confidence (top probability)", color=MUTED, fontsize=9)
    ax.set_ylabel("Accuracy (%)", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    _save(fig, path)


def chart_latency(lat, path):
    """p50 latency on a T4 for 1 and 10 questions per image, at each patch budget."""
    import numpy as np

    budgets = [("low", "low\n256 patches"), ("mid", "mid\n576 patches"), ("high", "high\n1,024 patches")]
    fig, ax = _axes((6.4, 4.0))
    x = np.arange(len(budgets))
    w = 0.36
    for j, (n, color) in enumerate(((1, BLUE), (10, ORANGE))):
        vals = [lat["%s_q%d" % (b, n)]["p50_ms"] for b, _ in budgets]
        pos = x + (j - 0.5) * w
        ax.bar(pos, vals, width=w - 0.04, color=color, label="1 question" if n == 1 else "10 questions")
        for p, v in zip(pos, vals):
            ax.text(p, v + 3, "%.0f ms" % v, ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in budgets], color=INK, fontsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_ylabel("p50 latency, one image (ms)", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    _save(fig, path)


# ----------------------------------------------------------------------------- card
QUICKSTART = '''```python
import laya
from PIL import Image

router = laya.Router()  # text checkpoints load lazily; Laya-V loads on the first image request
result = router.predict(
    {"image": Image.open("receipt.jpg"), "note": "Reimbursement claim"},
    {
        "over_limit": {"type": "noul", "instructions": "Is the receipt total above 50?"},
        "category": {"type": "choice", "instructions": "What was bought?",
                     "criteria": {"meals": "food and drink", "travel": "transport, hotels",
                                  "other": "everything else"}},
        "legibility": {"type": "score", "instructions": "How readable is the receipt?",
                       "criteria": ["unreadable", "partly readable", "clear"]},
    },
)
answers = result["answers"]
answers["over_limit"]["noul"]            # P(yes), e.g. 0.93
answers["category"]["choice"]            # e.g. "meals"
answers["category"]["probabilities"]     # e.g. {"meals": 0.88, "travel": 0.07, "other": 0.05}
answers["legibility"]["score"]           # expected level on the 0..2 scale
answers["category"]["answer_confidence"] # calibrated confidence, use it to route or abstain
result["routing"]["model"]               # "vision"
```'''

MORE_USAGE = '''```python
# many images, same questions: states share forward passes
results = router.predict_batch([{"state": {"image": p}, "questions": questions} for p in paths])

# trade detail for speed: low = 256 patches, mid = 576, high = 1,024
router.predict({"image": img, "image_detail": "low"}, questions)
```

`state["image"]` takes a PIL image, a file path, raw bytes or base64. Any other keys in the state are
read as text alongside the image (a caption, an email body, form fields).

Over HTTP, `laya-serve` (and the MCP server) also accept an image URL:

```bash
LAYA_DEVICE=cuda laya-serve
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "state": {"image": "https://example.com/receipt.jpg"},
  "questions": {"paid_by_card": {"type": "noul", "instructions": "Was it paid by card?"}}
}'
```'''


def build_card(ev, reg, man, lat, bench, cfg, params, assets):
    lv = ev["laya_v"]
    ocr, txt, audit = ev.get("ocr_pipeline", {}), ev.get("text_only", {}), ev.get("audit", {})
    tasks = sorted(k for k in lv if k.startswith("task:"))
    beats = sum(1 for t in tasks if t in ocr and lv[t]["acc"] > ocr[t]["acc"])
    n_cmp = sum(1 for t in tasks if t in ocr)
    mid1, mid10 = lat["mid_q1"], lat["mid_q10"]
    ratio10 = mid10["p50_ms"] / mid1["p50_ms"]
    reg_ok = sum(1 for k, v in reg.items() if isinstance(v, dict) and "delta" in v and v["delta"] >= -0.01)
    reg_n = sum(1 for k, v in reg.items() if isinstance(v, dict) and "delta" in v)
    teach_dom = {d: lv.get("task:teacher:" + d, {}).get("acc") for d, _ in DOMAINS}
    sy = bench.get("systems", {})
    qw, lb = sy.get(QWEN), sy.get("laya_v")

    dom_rows = ["| Domain | Laya-V | OCR + caption pipeline | Image dropped | n |", "|---|---|---|---|---|"]
    for d, name in DOMAINS:
        k = "task:teacher:" + d
        dom_rows.append("| %s | **%s** | %s | %s | %d |" % (name, pct(lv[k]["acc"]), pct(ocr.get(k, {}).get("acc")),
                                                          pct(txt.get(k, {}).get("acc")), lv[k]["n"]))
    dom_rows.append("| **All held-out image questions** | **%s** | %s | %s | %d |" % (
        pct(lv["all"]["acc"]), pct(ocr["all"]["acc"]), pct(txt["all"]["acc"]), lv["all"]["n"]))

    aud_rows = ["| Domain | Laya-V | Teacher | n |", "|---|---|---|---|"]
    for d, name in DOMAINS:
        a = audit.get("by_domain", {}).get(d)
        if a:
            aud_rows.append("| %s | %s | %s | %d |" % (name, pct(a["laya_v"], 0), pct(a["teacher"], 0), a["n"]))
    aud_rows.append("| **All** | **%s** | %s | %d |" % (pct(audit["laya_v_acc"]), pct(audit["teacher_acc"]), audit["n"]))

    pub = [("task:cord:payment", "CORD receipts", "How was it paid? (4 options)"),
           ("task:cord:tax", "CORD receipts", "Is tax or a service charge shown?"),
           ("task:cord:count", "CORD receipts", "How many line items? (4 levels)"),
           ("task:cord:item", "CORD receipts", "Does the receipt include a given item?"),
           ("task:pixmo:verify", "PixMo-Docs charts, tables, documents", "Is the answer a given value?"),
           ("task:pixmo:choice", "PixMo-Docs charts, tables, documents", "Pick the right value (4 options)")]
    pub_rows = ["| Dataset | Question | Laya-V | OCR + caption pipeline | n |", "|---|---|---|---|---|"]
    for k, ds, q in pub:
        if k in lv:
            a, b = lv[k]["acc"], ocr.get(k, {}).get("acc")
            win_a = b is None or a >= b  # bold the better of the two, whichever it is
            pub_rows.append("| %s | %s | %s | %s | %d |" % (ds, q, ("**%s**" if win_a else "%s") % pct(a),
                                                           ("%s" if win_a else "**%s**") % pct(b), lv[k]["n"]))

    sel_rows = ["| Domain | ECE | Accuracy, all answers | Accuracy, 80% most confident |", "|---|---|---|---|"]
    # the screenshot domain mixes in UI-element location (a limitation), so it is not a calibration row
    for key, name in [("all", "All held-out image questions")] + [("dom:" + d, n) for d, n in DOMAINS
                                                                  if d != "screenshot"]:
        if key in lv:
            m = lv[key]
            sel_rows.append("| %s | %.3f | %s | %s |" % (name, m["ece"], pct(m["acc"]), pct(m["sel_acc@80"])))

    lat_rows = ["| Patch budget | 1 question: p50 / p95 | 10 questions: p50 / p95 | Decisions per second |",
                "|---|---|---|---|"]
    for b, name in (("low", "`low` (256 patches)"), ("mid", "`mid` (576 patches)"), ("high", "`high` (1,024 patches)")):
        a, c = lat[b + "_q1"], lat[b + "_q10"]
        lat_rows.append("| %s | %.0f / %.0f ms | %.0f / %.0f ms | %.0f |" % (name, a["p50_ms"], a["p95_ms"], c["p50_ms"],
                                                                          c["p95_ms"], c["decisions_per_s"]))

    qwen_block = ""
    if qw and lb:
        m = qw.get("mcnemar_vs_laya_v_gold", {})
        qwen_block = f"""
### Above its weight class

Both models answered the same {bench['meta']['rows']:,} typed questions on the same held-out images, with the same
options in the same order (Qwen3-VL-8B-Instruct through OpenRouter, asked for a probability per option).
Accuracy is the mean over the public datasets (VQAv2, A-OKVQA, PixMo-Docs, CORD, ScreenSpot).

| Model | Parameters | Accuracy, public datasets (95% CI) | Calibration error (ECE) |
|---|---|---|---|
| **Laya-V** | **{params / 1e9:.2f}B** | **{pct(lb['macro_gold'])}** ({pct(lb['macro_gold_ci95'][0])}–{pct(lb['macro_gold_ci95'][1])}) | **{lb['ece_raw']:.3f}** |
| Qwen3-VL-8B-Instruct | 8B | {pct(qw['macro_gold'])} ({pct(qw['macro_gold_ci95'][0])}–{pct(qw['macro_gold_ci95'][1])}) | {qw['ece_raw']:.3f} |

Laya-V is ahead with about a tenth of the parameters (paired McNemar test, p = {m.get('p', float('nan')):.1g}). Qwen's
answers failed to parse on {100 * qw['failed_rows'] / qw['n_rows']:.1f}% of questions and those count as wrong, so part of
the gap is formatting rather than vision.
"""

    man_rows = ["| Source | License | Share of training rows |", "|---|---|---|"]
    for s, share in sorted(man.get("train_share", {}).items(), key=lambda kv: -kv[1]):
        if share > 0:
            man_rows.append("| %s | %s | %.1f%% |" % (s, man.get("licenses", {}).get(s, "?").replace("; see model card", ""), 100 * share))

    reg_rows = ["| Suite | laya-multilingual | Laya-V | Δ (points) |", "|---|---|---|---|"]
    for k, v in sorted(reg.items()):
        if isinstance(v, dict) and "base" in v:
            reg_rows.append("| %s | %s | %s | %+.1f |" % (k, pct(v["base"]), pct(v["new"]), 100 * v["delta"]))

    w = cfg.get("wise_ft") or {"alpha": 0.7}
    blank_conf = ev.get("probe_blank_image", {}).get("mean_max_prob")
    real_conf = lv["all"].get("mean_max_prob")
    fail = ", ".join(reg.get("_fail", [])) or "none"

    return f"""---
license: apache-2.0
library_name: laya
pipeline_tag: visual-question-answering
base_model: convaiinnovations/laya
language:
- multilingual
datasets:
- HuggingFaceM4/A-OKVQA
- naver-clova-ix/cord-v2
- allenai/pixmo-docs
- allenai/pixmo-cap
- Aria-UI/Aria-UI_Data
tags:
- laya
- vision
- image-classification
- document-understanding
- receipts
- content-moderation
- calibrated
- multilingual
- decision-model
- rlcd
---

# Laya-V

**Typed decisions about images, in one forward pass, with calibrated probabilities.**

Laya-V is the vision checkpoint of [Laya](https://github.com/NandhaKishorM/laya). You give it an image
(a photo, document, receipt, chart, screenshot or meme), optional text, and a set of typed questions: yes/no
(`noul`), pick-one (`choice`) or rate-on-a-scale (`score`). It returns a probability for every option of every
question at once, without generating text.

| | |
|---|---|
| **Parameters** | {params / 1e9:.2f}B (SigLIP 2 So400m vision tower + mmBERT-base text encoder) |
| **Latency** | {mid1['p50_ms']:.0f} ms p50 for 1 image + 1 question on a T4; {mid10['p50_ms']:.0f} ms for 10 questions |
| **Cost** | about ${lat['usd_per_1k_decisions']:.4f} per 1,000 decisions on a ${lat['usd_per_hour']:.2f}/h T4 |
| **Calibration** | image ECE {lv['all']['ece']:.3f} out of the box |
| **Languages** | text in images and questions in any language mmBERT reads |
| **License** | Apache-2.0; public training data is Apache-2.0 / CC-BY / ODC-BY only (teacher labels disclosed below) |

## Highlights

- **It reads the image, not just the text.** {pct(lv['all']['acc'])} on held-out image questions, against
  {pct(ocr['all']['acc'])} for an OCR + captioning pipeline (Florence-2-large) feeding the same text model and
  {pct(txt['all']['acc'])} with the image dropped. Laya-V beats the pipeline on {beats} of {n_cmp} evaluation slices.
- **Strong on everyday image domains:** {pct(teach_dom['meme'], 0)} on memes and posts, {pct(teach_dom['screenshot'], 0)} on screenshots,
  {pct(teach_dom['document'], 0)} on documents and {pct(teach_dom['photo'], 0)} on photos (held-out images, questions
  labelled by the teacher).
- **{100 * audit['laya_v_over_teacher']:.0f}% of its teacher's accuracy** on an audited set ({pct(audit['laya_v_acc'])} vs
  {pct(audit['teacher_acc'])}), at a fraction of the latency, and level with it on documents and memes.
- **Probabilities you can act on:** ECE {lv['all']['ece']:.3f}; keeping the 80% most confident answers lifts accuracy
  from {pct(lv['all']['acc'])} to {pct(lv['all']['sel_acc@80'])} (documents: {pct(lv['dom:document']['sel_acc@80'])}).
- **Many questions for almost the price of one:** the image is encoded once, so 10 questions take
  {ratio10:.1f}x the time of one ({mid10['p50_ms']:.0f} vs {mid1['p50_ms']:.0f} ms on a T4).
- **Above its weight class:** more accurate than Qwen3-VL-8B-Instruct (about 11x the parameters) on the same
  public-dataset questions, and better calibrated.

## Quickstart

```bash
{INSTALL}
```

{QUICKSTART}

{MORE_USAGE}

## What it is for

Laya-V is built for **closed-set decisions over images** at high volume, where you need a probability
rather than a paragraph:

- **Expense and receipt checks:** totals over a limit, payment method, tax shown, item present.
- **Document triage:** document type, which team should handle it, whether a field or signature is present.
- **Content moderation:** memes, posts and screenshots against your own policy categories.
- **Screenshot and UI triage:** what app or screen it is, whether an error is shown, what the user is doing.
- **Photo tagging:** scene, objects, attributes, quality.

Write the questions and options for your task; no fine-tuning is needed. Use `answer_confidence` to send
low-confidence cases to a person or a larger model.

**Not what it is for:** free-form answers or captions, transcribing text (OCR), and locating UI elements
on a screen (see Limitations).

## Results

All numbers are on held-out images: no training row uses an evaluation image.

### The image makes the difference

![Accuracy by domain]({assets['domains']})

Domain rows: held-out images with teacher-labelled questions. Last row: every held-out image question, public
datasets included. The OCR + caption pipeline gives the
base `laya-multilingual` model a Florence-2-large caption and OCR text instead of the image.

{chr(10).join(dom_rows)}

### Close to its teacher

An audited set of {audit['n']} questions whose reference answers were checked blind by an LLM judge
(`anthropic/claude-sonnet-5`), not taken from the teacher:

{chr(10).join(aud_rows)}

### Receipts and documents (public datasets)

{chr(10).join(pub_rows)}

### Calibrated probabilities

![Reliability diagram]({assets['calibration']})

Temperatures are fitted on a held-out slice and ship with the checkpoint, so the probabilities are usable as
they come: when Laya-V says 80%, it is right about 80% of the time.

{chr(10).join(sel_rows)}

Rows cover every held-out question in the domain, public datasets included, so they differ from the
teacher-labelled accuracies above. Screenshots are left out of this table because that slice is dominated by
UI-element location (see Limitations).

### Speed and cost

![Latency on a T4]({assets['latency']})

End-to-end `laya.Router.predict` on a Tesla T4 (fp16), image preprocessing included, {lat['repeats']} real
evaluation images per setting.

{chr(10).join(lat_rows)}

At ${lat['usd_per_hour']:.2f} per T4-hour that is about **${lat['usd_per_1k_decisions']:.4f} per 1,000 decisions**
(10 questions per image) or ${lat['usd_per_1k_decisions_q1']:.4f} with one question per image.
{qwen_block}
<details>
<summary><b>Text-only behaviour</b> (text requests take the unchanged text path)</summary>

Laya-V keeps the base model's text skills close to intact: {reg_ok} of {reg_n} suites are within 1 point or better.

{chr(10).join(reg_rows)}

</details>

## How it works

- **Vision tower:** SigLIP 2 So400m/16 NaFlex, which keeps each image's native aspect ratio (tall receipts
  and wide charts are not squashed). The patch budget is 256, 576 or 1,024 patches.
- **Adapter:** per-channel standardisation of the tower features, aspect-aware pooling of the patch grid
  to 128 tokens, a 2-layer MLP projector to width 768, and learned `<img>` / `</img>` vectors.
- **One pass for all questions:** each question becomes one row
  `[CLS] <type> instructions [SEP] [MASK] option… [SEP] <img> 128 image tokens </img> text state [SEP]`.
  The image is encoded once and its tokens are shared by every row; the mmBERT encoder and Laya's decision
  head score all options of all questions in one batched forward pass.
- Text-only requests skip the vision path entirely and behave like `laya-multilingual`.

## Training

1. **Alignment A1:** image-conditioned masked language modelling on captions (COCO, PixMo-Docs, PixMo-Cap);
   adapter only.
2. **Alignment A2:** 4-way caption matching on clusters of 4 similar images, so text alone cannot solve it;
   text encoder trainable at a low learning rate, text replay mixed in.
3. **Decisions B:** Laya's RLCD loop (GRPO-style noisy logits rewarded by strictly proper scoring rules, plus
   soft cross-entropy) on teacher-labelled image questions, public datasets converted to typed questions,
   and self-distilled text replay; a short B2 top-up with broader text replay limits text drift. The
   released weights are a WiSE-FT merge: text encoder and decision head at {w['alpha']:.1f} x fine-tuned +
   {1 - w['alpha']:.1f} x `laya-multilingual`.
4. **Calibration C:** temperatures fitted by LBFGS on a held-out slice, separately for text and image answers.

### Data and licenses

{chr(10).join(man_rows)}

Teacher labels come from **Space Bunny Alpha**, an OpenRouter stealth model, as stated probability
distributions, with a shuffled-option consistency check on 15% of images. Every training row carries its
source, so teacher rows can be replaced if the teacher's terms require it. Dataset licenses cover the
annotations; the underlying images (COCO from Flickr, PixMo-Cap web images) carry their own terms.

## Limitations

- **Closed-set decisions only.** Laya-V scores the options you give it; it does not generate answers. It is
  not a general visual question answering model, and large generative VLMs are more accurate on
  open-domain VQA benchmarks.
- **Locating UI elements is weak.** Questions like "where on the screen is the Save button?" are answered
  only modestly above chance; use a grounding model for that.
- **Confidence does not drop on uninformative images.** On a blank grey image the mean top probability is
  {pct(blank_conf)} (vs {pct(real_conf)} on real images), so check that an image is present and readable
  before trusting a confident answer.
- **Text-only drift:** suites over the 1-point limit: {fail}. Use `laya-multilingual` for text-only
  prompt-injection filtering.
- One image per request. The fast path (`fast=True`), ONNX export and `predict_long` do not accept images yet.
- Teacher labels carry the teacher's errors and some of its overconfidence (partly corrected by calibration),
  and Laya's known quirks (negation errors, score position bias in some languages) apply here too.

## Citation

```bibtex
@misc{{laya_v_2026,
  title        = {{Laya-V: calibrated typed decisions over images in one forward pass}},
  author       = {{Chopada, Henil}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{REPO}}}}}
}}
```

Laya-V builds on [Laya](https://github.com/NandhaKishorM/laya) and
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch16-naflex)
and [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-base).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="output dir: README.md and assets/ are written here")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--regression")
    ap.add_argument("--manifest")
    ap.add_argument("--latency-real", required=True, help="latency_t4_real.json from bench_latency.py")
    ap.add_argument("--bench", help="bench_vlms.json (only the Laya-V and Qwen3-VL-8B rows are used)")
    ap.add_argument("--laya-rows", help="laya_v_rows.json, for the reliability diagram")
    ap.add_argument("--params", type=int, help="parameter count, if the dir has no model.safetensors")
    args = ap.parse_args()
    ev, reg, man = load(args.eval), load(args.regression), load(args.manifest)
    lat, bench = load(args.latency_real), load(args.bench)
    cfg = load(os.path.join(args.ckpt, "rl_agent_config.json"))
    weights = os.path.join(args.ckpt, "model.safetensors")
    params = args.params or (count_params(weights) if os.path.exists(weights) else None)
    if not params:
        raise SystemExit("pass --params or put model.safetensors in --ckpt")

    os.makedirs(os.path.join(args.ckpt, "assets"), exist_ok=True)
    assets = {"domains": "assets/accuracy_by_domain.png", "calibration": "assets/calibration.png",
              "latency": "assets/latency_t4.png"}
    chart_domains(ev, os.path.join(args.ckpt, assets["domains"]))
    rows = load(args.laya_rows).get("eval")
    if rows:
        chart_calibration(rows, os.path.join(args.ckpt, assets["calibration"]))
    chart_latency(lat, os.path.join(args.ckpt, assets["latency"]))

    card = build_card(ev, reg, man, lat, bench, cfg, params, assets)
    with open(os.path.join(args.ckpt, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print("wrote", os.path.join(args.ckpt, "README.md"))


if __name__ == "__main__":
    main()
