#!/usr/bin/env bash
# Laya-V vs VLMs on a rented T4: setup, eval images, Laya-V latency + accuracy, OpenRouter VLMs.
#   HF_TOKEN=... OPENROUTER_API_KEY=... USD_PER_HOUR=0.20 bash research/vision/t4_bench.sh [setup|latency|bench|all]
# Items come from the Hub (henilchopada/laya-v-data built_items), the same files the A100 eval used.
# Public eval images are not on the Hub: COCO val2017 is downloaded, and the A-OKVQA / PixMo-Docs / CORD /
# ScreenSpot builders are re-run only to write their images (their items go to *_t4 files and are unused).
set -euo pipefail
W=${W:-/workspace}
DATA=$W/data
RAW=$W/raw
CK=$W/ckpt/laya-v
OUT=$W/runs/bench
PY=research/vision
mkdir -p "$DATA" "$RAW" "$W/ckpt" "$OUT" "$W/logs"
export HF_XET_HIGH_PERFORMANCE=1

setup() {
  pip install -q -U "transformers>=5.0" safetensors huggingface_hub datasets pyarrow pillow httpx tokenizers \
      einops matplotlib
  pip install -q -e . --no-deps
  python -c "import torch, transformers; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'tf', transformers.__version__)"
  hf download henilchopada/laya-v --local-dir "$CK"
  hf download google/siglip2-so400m-patch16-naflex
  hf download henilchopada/laya-v-data --repo-type dataset --local-dir "$DATA" --include "images/*" "built_items/stage_b.jsonl" \
      "built_items/audit_items.jsonl" "built_items/audit_verdicts_sonnet5.json" "built_items/captions_coco.jsonl"
  mkdir -p "$DATA/pool" "$DATA/items"
  [ -e "$DATA/pool/images" ] || ln -s "$DATA/images" "$DATA/pool/images"   # teacher items say pool/images/...
  cp "$DATA"/built_items/*.jsonl "$DATA"/built_items/*.json "$DATA/items/"
  # COCO val2017 only (VQAv2 and A-OKVQA eval images)
  if [ ! -d "$DATA/coco/val2017" ]; then
    (cd "$RAW" && curl -sSfLO http://images.cocodataset.org/zips/val2017.zip && mkdir -p "$DATA/coco" \
      && unzip -q -o val2017.zip -d "$DATA/coco" && rm -f val2017.zip)
  fi
  hf download HuggingFaceM4/A-OKVQA --repo-type dataset --local-dir "$RAW/aokvqa"
  hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir "$RAW/cord"
  hf download rootsautomation/ScreenSpot --repo-type dataset --local-dir "$RAW/screenspot"
  hf download allenai/pixmo-docs --repo-type dataset --local-dir "$RAW/pixmo-docs" --include "*/validation-*"
  python $PY/build_public.py --data "$DATA" --raw "$RAW" --only aokvqa pixmo cord screenspot --tag _t4
  # every eval image the benchmark will use must exist
  python - "$DATA" <<'EOF'
import json, os, sys
d = sys.argv[1]
miss, n = [], 0
for p in ("items/stage_b.jsonl", "items/audit_items.jsonl"):
    for line in open(os.path.join(d, p), encoding="utf-8"):
        r = json.loads(line)
        if r.get("image") and (r.get("split") == "eval" or p.startswith("items/audit")):
            n += 1
            if not os.path.exists(os.path.join(d, r["image"])):
                miss.append(r["image"])
print("eval rows with images", n, "missing", len(miss), sorted(set(miss))[:10])
sys.exit(1 if miss else 0)
EOF
}

latency() {
  python $PY/bench_latency.py --ckpt "$CK" --items "$DATA/items/stage_b.jsonl" --image-root "$DATA" \
      --usd-per-hour "${USD_PER_HOUR:-0.20}" --out "$OUT/latency_t4_real.json" 2>&1 | tee "$W/logs/latency.log"
}

bench() {
  : "${OPENROUTER_API_KEY:?export OPENROUTER_API_KEY}"
  # the teacher first: its free preview ends around 29 Sep 2026
  python $PY/bench_vlms.py --items "$DATA/items/stage_b.jsonl" --image-root "$DATA" --out "$OUT" \
      --final "$CK" --audit-items "$DATA/items/audit_items.jsonl" \
      --audit-verdicts "$DATA/items/audit_verdicts_sonnet5.json" --max-usd "${MAX_USD:-30}" 2>&1 | tee "$W/logs/bench.log"
}

case "${1:-all}" in
  setup) setup ;;
  latency) latency ;;
  bench) bench ;;
  all) setup; latency; bench ;;
esac
