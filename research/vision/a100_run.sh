#!/usr/bin/env bash
# Laya-V A100 runbook. Each step is a function; run them in order and check each gate:
#   bash research/vision/a100_run.sh public stage_a_items replay stage_a mix stage_b calibrate regression eval
set -euo pipefail
W=${W:-/workspace}
D=$W/data
R=$W/raw
CK=$W/ckpt/laya/multilingual
RUNS=$W/runs
export LAYA_V_HUB_REPO=${LAYA_V_HUB_REPO:-henilchopada/laya-v}
export TOKENIZERS_PARALLELISM=false
PY="python -u research/vision"

public() {        # public datasets -> images + items (pixmo-cap downloads run alongside)
  $PY/build_public.py --data $D --raw $R --only pixcap > $W/logs/public_pixcap.log 2>&1 &
  $PY/build_public.py --data $D --raw $R --only coco aokvqa cord screenspot > $W/logs/public_a.log 2>&1 &
  $PY/build_public.py --data $D --raw $R --only pixmo > $W/logs/public_pixmo.log 2>&1 &
  $PY/build_public.py --data $D --raw $R --only aria > $W/logs/public_aria.log 2>&1 &
  wait
  grep -h DONE $W/logs/public_*.log
}
stage_a_items() { $PY/build_stage_a.py --data $D --max-train 300000; }
overfit() {       # stage 0 gate: 64 image groups must be learned (near 100% agreement)
  $PY/train.py --stage B --base $CK --items $D/items/teacher_items.jsonl --image-root $D --out $RUNS/overfit \
    --overfit 64 --max-steps 200 --rows 64 --accum 1 --eval-every 50 --lr-encoder 5e-5 --lr-head 3e-4 --lr-adapter 1e-3
}
stage_a() {       # A1: image-conditioned MLM alignment (adapter only; dense signal per masked caption token)
  $PY/train_mlm.py --base $CK --data $D --out $RUNS/mlm --steps 2000 --eval-every 400 --batch 128
  # A2: caption matching on symmetric 4-image clusters, encoder trainable, text replay mixed in.
  # Gate: shuffled-image probe must collapse to chance while real images score high.
  $PY/train.py --stage A --init $RUNS/mlm/final --items $D/items/stage_a.jsonl $D/items/replay.jsonl     --image-root $D --out $RUNS/stage_a2 --epochs 0.5 --rows 128 --accum 2 --eval-every 250 --eval-batches 40     --train-encoder-a --lr-resampler-a 2e-4 --lr-projector-a 2e-4
}
replay() { $PY/build_replay.py --data $D --raw $R --base $CK --n 40000; }
mix() {           # stage B mixture, rows with missing images dropped, 40k caption rows kept for alignment
  $PY/mix_stage_b.py --data $D
  python - <<PYEOF
import json, os, random
d = "$D/items"
for f in ["stage_b"]:
    rows = [l for l in open(os.path.join(d, f + ".jsonl")) if not (json.loads(l).get("image") and not os.path.exists("$D/" + json.loads(l)["image"]))]
    open(os.path.join(d, f + ".jsonl"), "w").writelines(rows)
keep = [l for l in open(os.path.join(d, "stage_a.jsonl")) if '"split": "train"' in l]
random.Random(3).shuffle(keep)
open(os.path.join(d, "stage_a_keep.jsonl"), "w").writelines(keep[:40000])
PYEOF
}
stage_b() {
  $PY/train.py --stage B --init $RUNS/stage_a2/final --items $D/items/stage_b.jsonl $D/items/stage_a_keep.jsonl     --image-root $D --out $RUNS/stage_b --epochs ${EPOCHS:-2} --rows 32 --accum 2 --eval-every 1000 --eval-batches 60
}
calibrate() { $PY/calibrate.py --ckpt $RUNS/stage_b/final --items $D/items/stage_b.jsonl --image-root $D; }
regression() { $PY/text_regression.py --base $CK --new $RUNS/stage_b/final --out $RUNS/text_regression.json; }
eval() {
  $PY/eval_vision.py --final $RUNS/stage_b/final --stage-a $RUNS/stage_a2/final --base $CK \
    --items $D/items/stage_b.jsonl --image-root $D --out $RUNS/eval \
    ${AUDIT:+--audit $AUDIT --audit-items $D/items/audit_items.jsonl}
}

mkdir -p $W/logs $RUNS
for step in "$@"; do
  echo "=== $step $(date)"
  $step
done
