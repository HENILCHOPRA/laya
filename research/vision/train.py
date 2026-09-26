"""Laya-V training, single GPU (A100). Stage A (alignment) and stage B (RLCD decisions).

Stage A trains only the vision adapter (resampler, projector, boundary vectors) with soft
cross-entropy on caption-matching questions; the tower, mmBERT and the decision head are frozen.
Stage B trains everything but the tower with the fine-tuning notebook's loss: GRPO-style
policy gradient over G noisy logit samples rewarded by `proper_reward`, plus 1.0 x soft CE.

    # stage A from laya-multilingual
    python research/vision/train.py --stage A --base $CK/multilingual --items data/items/stage_a.jsonl \
        --image-root data/images --out runs/stage_a
    # stage B from stage A
    python research/vision/train.py --stage B --init runs/stage_a/final --items data/items/stage_b.jsonl \
        --image-root data/images --out runs/stage_b
    # overfit test (stage 0)
    python research/vision/train.py --stage B --base $CK/multilingual --items data/items/overfit.jsonl \
        --image-root data/images --out runs/overfit --overfit 64 --max-steps 200
"""
import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vdata  # noqa: E402  (also puts the repo root on sys.path)
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import build_model, ece_score, proper_reward  # noqa: E402
from laya.vision import build_vision_adapter, vision_config_for  # noqa: E402

DEFAULT_VISION = "google/siglip2-so400m-patch16-naflex"


# ----------------------------------------------------------------------------- model io
def load_base(base_dir, vision_encoder, device, query_init_std=0.02):
    """laya-multilingual + a fresh vision adapter (tower from the Hub)."""
    from safetensors.torch import load_file

    _fix_tokenizer_config(base_dir)
    with open(os.path.join(base_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(base_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(base_dir, "model.safetensors")), strict=True)
    adapter = build_vision_adapter(vision_config_for(vision_encoder, {}, query_init_std=query_init_std),
                                   model.encoder.config.hidden_size, pretrained=True)
    model.vision = adapter
    cfg["vision"] = adapter.vcfg
    return model, cfg


def load_laya_v(ckpt_dir):
    from safetensors.torch import load_file

    with open(os.path.join(ckpt_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(ckpt_dir, "encoder"), pretrained=False)
    model.load_state_dict(load_file(os.path.join(ckpt_dir, "model.safetensors")), strict=True)
    return model, cfg


def load_tokenizer(ckpt_dir):
    from transformers import AutoTokenizer

    _fix_tokenizer_config(ckpt_dir)
    return AutoTokenizer.from_pretrained(os.path.join(ckpt_dir, "tokenizer"))


def save_laya_v(model, cfg, tok, out_dir, extra_cfg=None):
    """A checkpoint `laya.Agent` loads directly (strict load, vision tower included)."""
    from safetensors.torch import save_file

    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    sd = {k: v.detach().to(torch.float16).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, os.path.join(tmp, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(tmp, "encoder"))
    tok.save_pretrained(os.path.join(tmp, "tokenizer"))
    c = dict(cfg)
    c["vision"] = model.vision.vcfg
    c.update(extra_cfg or {})
    with open(os.path.join(tmp, "rl_agent_config.json"), "w") as f:
        json.dump(c, f, indent=2)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def push_hub(local_dir, repo, path_in_repo, token=None):
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo, path_in_repo=path_in_repo,
                      commit_message="laya-v %s" % path_in_repo)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, return_rows=False):
    """Accuracy (argmax vs target argmax), soft accuracy, ECE, per task and source."""
    model.eval()
    per = defaultdict(lambda: {"n": 0, "correct": 0.0, "soft": 0.0})
    confs, corrects, rows_out = [], [], []
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = vdata.model_forward(model, b, device)
        logits = logits.float().cpu()
        for i, meta in enumerate(b["meta"]):
            k = int(b["marker_mask"][i].sum())
            z = logits[i, :k].numpy()
            p = vdata.softmax_np(z)
            t = b["target"][i, :k].numpy()
            ok = float(p.argmax() == t.argmax())
            for key in ("all", "task:%s" % meta.get("task"), "src:%s" % meta.get("source"),
                        "dom:%s" % meta.get("domain"), "type:%d" % int(b["qtype"][i])):
                d = per[key]
                d["n"] += 1
                d["correct"] += ok
                d["soft"] += float((p * t).sum())
            confs.append(float(p.max()))
            corrects.append(ok)
            if return_rows:
                rows_out.append({"meta": meta, "logits": z.tolist(), "target": t.tolist(), "qtype": int(b["qtype"][i])})
    res = {k: {"n": v["n"], "acc": v["correct"] / max(1, v["n"]), "soft_acc": v["soft"] / max(1, v["n"])}
           for k, v in sorted(per.items())}
    res["ece"] = ece_score(np.array(confs), np.array(corrects)) if confs else float("nan")
    model.train()
    return (res, rows_out) if return_rows else res


# ----------------------------------------------------------------------------- training
def set_trainable(model, stage, train_encoder=False):
    for p in model.parameters():
        p.requires_grad_(False)
    adapter = [model.vision.resampler, model.vision.projector]
    for m in adapter:
        for p in m.parameters():
            p.requires_grad_(True)
    model.vision.img_start.requires_grad_(True)
    model.vision.img_end.requires_grad_(True)
    if stage == "B":
        for name, p in model.named_parameters():
            if not name.startswith("vision."):
                p.requires_grad_(True)
    elif train_encoder:
        # Stage A with the text encoder adapting at a low rate (decision head stays frozen).
        for p in model.encoder.parameters():
            p.requires_grad_(True)
    model.vision.freeze_tower()


def param_groups(model, args):
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("vision.resampler"):
            groups["resampler"].append(p)
        elif name.startswith("vision."):
            groups["projector"].append(p)  # projector + boundary vectors
        elif name.startswith("encoder."):
            groups["encoder"].append(p)
        else:
            groups["head"].append(p)
    lr = {"A": {"resampler": args.lr_resampler_a, "projector": args.lr_projector_a, "encoder": args.lr_encoder_a},
          "B": {"resampler": args.lr_adapter, "projector": args.lr_adapter, "encoder": args.lr_encoder,
                "head": args.lr_head}}[args.stage]
    return [{"params": ps, "lr": lr[g], "name": g} for g, ps in groups.items()]


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return max(0.02, 0.5 * (1 + math.cos(math.pi * min(1.0, prog))))


def loss_fn(logits, b, device, stage, sigma, group_size, w_sph, w_rps):
    logits = logits.float()
    mask = b["marker_mask"].to(device)
    target = b["target"].to(device)
    logp_all = torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
    loss_ce = -(target * logp_all).sum(-1).mean()
    if stage == "A":
        return loss_ce, {"ce": loss_ce.item()}
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
    loss_rl = -(adv * logp).mean()
    return loss_rl + 1.0 * loss_ce, {"ce": loss_ce.item(), "rl": loss_rl.item(), "reward": r.mean().item()}


def make_loader(groups, tok, args, shuffle, seed=0, max_rows=None):
    ds = vdata.GroupDataset(groups, tok, args.image_root, args.max_len, args.head_max_len)
    sampler = vdata.BudgetBatchSampler(ds, max_rows or args.rows, shuffle=shuffle, seed=seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None,
                        collate_fn=lambda g: vdata.collate_groups(g, tok.pad_token_id))
    return loader, sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--base", help="laya-multilingual checkpoint dir (fresh adapter)")
    ap.add_argument("--init", help="Laya-V checkpoint dir to continue from")
    ap.add_argument("--vision-encoder", default=DEFAULT_VISION)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--rows", type=int, default=64, help="question rows per micro-batch")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-adapter", type=float, default=2e-4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--sigma-start", type=float, default=0.4)
    ap.add_argument("--sigma-end", type=float, default=0.1)
    ap.add_argument("--w-sph", type=float, default=0.75)
    ap.add_argument("--w-rps", type=float, default=1.0)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=60)
    ap.add_argument("--save-every-min", type=float, default=30)
    ap.add_argument("--hub-repo", default=os.environ.get("LAYA_V_HUB_REPO"))
    ap.add_argument("--hub-every-min", type=float, default=120)
    ap.add_argument("--overfit", type=int, default=0, help="train and evaluate on the first N groups only")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--stat-batches", type=int, default=40)
    ap.add_argument("--lr-resampler-a", type=float, default=3e-4)
    ap.add_argument("--lr-projector-a", type=float, default=1e-3)
    ap.add_argument("--query-init-std", type=float, default=0.02)
    ap.add_argument("--train-encoder-a", action="store_true", help="stage A: also train the text encoder")
    ap.add_argument("--lr-encoder-a", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    latest = os.path.join(args.out, "latest")
    resume = os.path.exists(os.path.join(latest, "train_state.pt"))
    if resume:
        model, cfg = load_laya_v(latest)
        tok = load_tokenizer(latest)
    elif args.init:
        model, cfg = load_laya_v(args.init)
        tok = load_tokenizer(args.init)
    else:
        model, cfg = load_base(args.base, args.vision_encoder, device, args.query_init_std)
        tok = load_tokenizer(args.base)
    cfg["max_len"], cfg["head_max_len"] = args.max_len, args.head_max_len
    model.to(device)
    if device.type == "cuda":
        model.vision.tower.to(torch.bfloat16)
    set_trainable(model, args.stage, args.train_encoder_a)
    if not args.no_grad_ckpt:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.train()
    if args.stage == "A" and not args.train_encoder_a:
        # Frozen text side: no dropout there, only the adapter trains.
        model.encoder.eval()
        if model.head is not None:
            model.head.eval()

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("stage %s | trainable params %.1fM" % (args.stage, n_train / 1e6), flush=True)

    train_items = vdata.read_items(args.items, "train")
    eval_items = vdata.read_items(args.items, "eval")
    groups = vdata.group_items(train_items)
    if args.overfit:
        random.Random(0).shuffle(groups)
        groups = groups[: args.overfit]
        eval_groups = groups
    else:
        eval_groups = vdata.group_items(eval_items)
        random.Random(1).shuffle(eval_groups)
        eval_groups = eval_groups[: args.eval_batches * max(1, args.rows // 8)]
    print("train groups %d rows %d | eval groups %d" % (len(groups), sum(len(g) for g in groups), len(eval_groups)),
          flush=True)
    loader, sampler = make_loader(groups, tok, args, shuffle=True, seed=args.seed)
    if not resume and not args.init:
        # Fresh adapter: measure tower-feature statistics before the first step.
        from laya.vision import fit_feature_stats
        stat_batches = []
        for bi, b in enumerate(loader):
            if b["pixels"] is not None:
                stat_batches.append(b["pixels"])
            if len(stat_batches) >= args.stat_batches:
                break
        n = fit_feature_stats(model.vision, stat_batches, device)
        print("feature stats from %d patches" % n, flush=True)
        del stat_batches
    eval_loader = make_loader(eval_groups, tok, args, shuffle=False)[0] if eval_groups else None

    steps_per_epoch = math.ceil(len(sampler) / args.accum)
    total = args.max_steps or int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(param_groups(model, args), weight_decay=0.01, betas=(0.9, 0.98))
    base_lrs = [g["lr"] for g in opt.param_groups]
    step, epoch, skip = 0, 0, 0
    if resume:
        st = torch.load(os.path.join(latest, "train_state.pt"), map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        step, epoch, skip = st["step"], st["epoch"], st["batch_in_epoch"]
        print("resumed at step %d epoch %d batch %d" % (step, epoch, skip), flush=True)
    warmup = int(total * args.warmup)
    log = open(os.path.join(args.out, "train_log.jsonl"), "a")
    t_save = t_hub = time.time()
    t0 = time.time()
    rows_seen = 0
    best = None

    def checkpoint(tag_dir, with_state, bi):
        save_laya_v(model, cfg, tok, tag_dir, {"laya_v_stage": args.stage, "laya_v_step": step})
        if with_state:
            torch.save({"opt": opt.state_dict(), "step": step, "epoch": epoch, "batch_in_epoch": bi},
                       os.path.join(tag_dir, "train_state.pt"))

    done = False
    while not done:
        sampler.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        micro = 0
        for bi, b in enumerate(loader):
            if bi < skip:
                continue
            progress = step / max(1, total)
            sigma = args.sigma_start + (args.sigma_end - args.sigma_start) * progress
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, act = vdata.model_forward(model, b, device)
            loss, parts = loss_fn(logits, b, device, args.stage, sigma, args.group_size, args.w_sph, args.w_rps)
            (loss / args.accum + 0.0 * act.float().sum()).backward()
            micro += 1
            rows_seen += b["input_ids"].shape[0]
            if micro % args.accum:
                continue
            scale = cosine_warmup(step, total, warmup)
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * scale
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                rec = {"step": step, "total": total, "epoch": epoch, "loss": loss.item(), **parts,
                       "lr_scale": scale, "sigma": sigma, "rows_per_s": rows_seen / el, "elapsed_min": el / 60,
                       "eta_min": el / 60 / max(1, step - (st["step"] if resume else 0)) * (total - step)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if eval_loader is not None and (step % args.eval_every == 0 or step == total):
                res = evaluate(model, eval_loader, device, None if args.overfit else args.eval_batches)
                if args.stage == "A" and not args.train_encoder_a:
                    model.encoder.eval()
                    if model.head is not None:
                        model.head.eval()
                summary = {k: round(v["acc"], 4) for k, v in res.items() if isinstance(v, dict) and
                           (k == "all" or k.startswith("task:") or k.startswith("dom:"))}
                print(json.dumps({"eval_step": step, "ece": res["ece"], **summary}), flush=True)
                log.write(json.dumps({"eval_step": step, "eval": res}) + "\n")
                log.flush()
            now = time.time()
            if now - t_save > args.save_every_min * 60:
                checkpoint(latest, True, bi + 1)
                t_save = now
                if args.hub_repo and now - t_hub > args.hub_every_min * 60:
                    push_hub(latest, args.hub_repo, "runs/%s/latest" % os.path.basename(args.out.rstrip("/")))
                    t_hub = now
            if step >= total:
                done = True
                break
        else:
            epoch += 1
            skip = 0
            continue
        break

    final = os.path.join(args.out, "final")
    checkpoint(final, False, 0)
    if eval_loader is not None:
        res = evaluate(model, eval_loader, device, None)
        with open(os.path.join(args.out, "final_eval.json"), "w") as f:
            json.dump(res, f, indent=1)
        print("FINAL", json.dumps({k: round(v["acc"], 4) for k, v in res.items() if isinstance(v, dict)}), flush=True)
    if args.hub_repo:
        push_hub(final, args.hub_repo, "runs/%s/final" % os.path.basename(args.out.rstrip("/")))
    print("done in %.1f min" % ((time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
    # Persistent DataLoader workers can keep the interpreter from exiting; everything is saved.
    sys.stdout.flush()
    os._exit(0)
