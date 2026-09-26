# Laya-V pipeline

This folder builds Laya-V, the image checkpoint. The library-side code is in `laya/vision.py`, plus
`build_sequence(image_slots=...)`, `DecisionModel.forward(vis_tokens=...)`, and image states in
`Agent`, `Router`, `laya-serve` and MCP.

| Step | Where | Script |
| --- | --- | --- |
| Image pool for the teacher (about 22k images, 6 domains) | laptop | `fetch_label_pool.py`, `make_memes.py` |
| Question bank: paraphrases and translations | laptop | `question_bank.py`, `build_question_bank.py` |
| Teacher labels (10 typed questions per image, 15% order check) | laptop | `teacher_label.py`, `openrouter.py` |
| Teacher labels to items; audit sample and page | laptop | `build_teacher_items.py`, `build_audit_page.py` |
| Latency gate on the target GPU | Colab T4 | `latency_gate.py` |
| Environment and downloads | A100 | `a100_setup.sh` |
| Public datasets to items; Stage A captions | A100 | `build_public.py` |
| Stage A caption matching with hard negatives | A100 | `build_stage_a.py` |
| Text replay, self-distilled | A100 | `build_replay.py` |
| Stage B mixture and manifest | A100 | `mix_stage_b.py` |
| Training (overfit test, Stage A, Stage B) | A100 | `train.py`, `vdata.py` |
| Calibration (Stage C) | A100 | `calibrate.py` |
| Text regression gate | A100 | `text_regression.py` |
| Evaluation, baselines, probes, audit | A100 | `eval_vision.py` |

`a100_run.sh` runs the A100 steps in order, one named step at a time.

## Item format

Items are JSONL files with one row per question. Rows that share a `gid` share one image and one
pass of the vision tower. See `vdata.py`.
