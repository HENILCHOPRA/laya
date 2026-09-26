#!/usr/bin/env bash
# Laya-V A100 box setup: environment, base checkpoint, vision tower, all datasets (in parallel).
#   bash research/vision/a100_setup.sh            # from the repo root, HF_TOKEN exported
set -euo pipefail
W=${W:-/workspace}
DATA=$W/data
RAW=$W/raw
mkdir -p "$DATA" "$RAW" "$W/ckpt" "$W/runs" "$W/logs"
export HF_XET_HIGH_PERFORMANCE=1

[ -n "${SKIP_PIP:-}" ] || pip install -q -U "transformers>=5.0" safetensors huggingface_hub datasets pyarrow pillow httpx \
    tokenizers ijson accelerate einops
[ -n "${SKIP_PIP:-}" ] || pip install -q -e . --no-deps
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'tf', transformers.__version__)"

dl() {  # dl <name> <command...> : run in background, log to $W/logs/<name>.log
  local name=$1; shift
  ( "$@" > "$W/logs/dl_$name.log" 2>&1 && echo "OK" >> "$W/logs/dl_$name.log" || echo "FAILED" >> "$W/logs/dl_$name.log" ) &
}

# model weights
dl base    hf download convaiinnovations/laya --include "multilingual/*" --local-dir "$W/ckpt/laya"
dl siglip  hf download google/siglip2-so400m-patch16-naflex
dl siglipb hf download google/siglip2-base-patch16-naflex

# teacher pool + items (uploaded from the labelling laptop)
dl teacher hf download henilchopada/laya-v-data --repo-type dataset --local-dir "$DATA" --max-workers 64

# COCO 2017 (+ annotations) and VQAv2 annotations
dl coco bash -c "cd $RAW && for f in zips/train2017.zip zips/val2017.zip annotations/annotations_trainval2017.zip; do \
   curl -sSfLO http://images.cocodataset.org/\$f; done && mkdir -p $DATA/coco && \
   unzip -q -o train2017.zip -d $DATA/coco && unzip -q -o val2017.zip -d $DATA/coco && \
   unzip -q -o annotations_trainval2017.zip -d $DATA/coco && rm -f train2017.zip val2017.zip"
dl vqa bash -c "mkdir -p $RAW/vqa && cd $RAW/vqa && for f in v2_Questions_Train_mscoco v2_Annotations_Train_mscoco \
   v2_Questions_Val_mscoco v2_Annotations_Val_mscoco; do curl -sSfLO https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/\$f.zip \
   && unzip -q -o \$f.zip; done"

# HF datasets (parquet)
dl aokvqa hf download HuggingFaceM4/A-OKVQA --repo-type dataset --local-dir "$RAW/aokvqa"
dl cord   hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir "$RAW/cord"
dl sspot  hf download rootsautomation/ScreenSpot --repo-type dataset --local-dir "$RAW/screenspot"
dl pixcap hf download allenai/pixmo-cap --repo-type dataset --local-dir "$RAW/pixmo-cap"
# pixmo-docs: validation of every subset + a subset of train shards (~60k images)
dl pixdocs hf download allenai/pixmo-docs --repo-type dataset --local-dir "$RAW/pixmo-docs" \
   --include "*/validation-*" "charts/train-0000[0-9]-*" "charts/train-0001[0-5]-*" "tables/train-0000[0-5]-*" \
             "diagrams/train-*" "other/train-0000[0-9]-*" "other/train-0001[0-5]-*"
# Aria-UI desktop screenshots + element annotations with instructions
dl aria hf download Aria-UI/Aria-UI_Data --repo-type dataset --local-dir "$RAW/aria" \
   desktop/screenshots.zip desktop/aria_ui_desktop_with_instructions.json

wait
grep -H -E "OK|FAILED" "$W"/logs/dl_*.log
