"""Laya-V evaluation: systems x slices, calibration, selective accuracy, failure probes, audit.

Systems, all on the same eval subset (<= --per-task rows per task, grouped by image):
    laya_v        final checkpoint (calibrated temperatures applied)
    laya_v_A      stage A checkpoint (aligned, not decision-tuned), if --stage-a is given
    text_only     base laya-multilingual, image dropped (floor: what the image adds)
    ocr_pipeline  Florence-2-large caption + OCR text in the state, base laya-multilingual (approach 1)
    teacher       the VLM teacher on a smaller subsample (--teacher-per-task), ceiling
Probes (laya_v): shuffled images, contradicting caption, blank grey image.
Audit: if --audit verdicts are given, accuracy of teacher and Laya-V against the human labels.

    python research/vision/eval_vision.py --final runs/stage_b/final --stage-a runs/stage_a/final \
        --base /workspace/ckpt/laya/multilingual --items data/items/stage_b.jsonl --image-root data \
        --out runs/eval
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402
import vdata  # noqa: E402
from calibrate import temp_for  # noqa: E402
from laya.common import ece_score  # noqa: E402


# ----------------------------------------------------------------------------- metrics
def metrics(rows):
    """rows: [{"p": probs, "target": [...], "qtype": int, "task": str, "domain": str}]"""
    out = {}
    by = defaultdict(list)
    for r in rows:
        for key in ("all", "task:" + r["task"], "dom:" + r["domain"], "type:%d" % r["qtype"]):
            by[key].append(r)
    for key, rs in sorted(by.items()):
        p = [np.array(r["p"]) for r in rs]
        t = [np.array(r["target"]) for r in rs]
        correct = np.array([float(a.argmax() == b.argmax() and not r.get("failed")) for a, b, r in zip(p, t, rs)])
        conf = np.array([a.max() for a in p])
        onehot = [np.eye(len(b))[b.argmax()] for b in t]
        m = {"n": len(rs), "acc": float(correct.mean()),
             "soft_acc": float(np.mean([(a * b).sum() for a, b in zip(p, t)])),
             "brier": float(np.mean([((a - o) ** 2).sum() for a, o in zip(p, onehot)])),
             "ece": float(ece_score(conf, correct))}
        sc = [(float((np.arange(len(a)) * a).sum()), float(b.argmax())) for a, b, r in zip(p, t, rs) if r["qtype"] == 1]
        if sc:
            m["score_mae"] = float(np.mean([abs(x - y) for x, y in sc]))
        order = np.argsort(-conf)
        for cov in (0.8, 0.9):
            k = max(1, int(len(order) * cov))
            m["sel_acc@%d" % int(cov * 100)] = float(correct[order[:k]].mean())
        out[key] = m
    return out


def softmax(z):
    return vdata.softmax_np(np.asarray(z, dtype=np.float64))


# ----------------------------------------------------------------------------- systems
def laya_v_rows(ckpt, groups, args, image_override=None, state_override=None):
    model, cfg = T.load_laya_v(ckpt)
    tok = T.load_tokenizer(ckpt)
    model.to("cuda").eval()
    model.vision.tower.to(torch.bfloat16)
    if state_override:
        groups = [[dict(r, state=state_override(r)) for r in g] for g in groups]
    ds = vdata.GroupDataset(groups, tok, args.image_root, args.max_len, args.head_max_len,
                            image_override=image_override)
    sampler = vdata.BudgetBatchSampler(ds, args.rows, shuffle=False)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=args.workers,
                                         collate_fn=lambda g: vdata.collate_groups(g, tok.pad_token_id))
    _, rows = T.evaluate(model, loader, torch.device("cuda"), None, return_rows=True)
    out = []
    for r in rows:
        img = r["meta"]["domain"] != "text"
        t = temp_for(cfg, r["qtype"], len(r["logits"]), img)
        out.append({"p": softmax(np.array(r["logits"]) / t).tolist(), "target": r["target"], "qtype": r["qtype"],
                    "task": r["meta"]["task"], "domain": r["meta"]["domain"], "gid": r["meta"]["gid"],
                    "qkey": r["meta"].get("qkey")})
    del model
    torch.cuda.empty_cache()
    return out


def agent_rows(agent, groups, state_fn):
    """Run a text checkpoint through laya.Agent with a per-group text state."""
    from laya.common import QTYPES
    out = []
    for g in groups:
        state = state_fn(g)
        qs = {"q%d" % i: r["q"] for i, r in enumerate(g)}
        try:
            ans = agent.predict(state, qs)["answers"]
        except ValueError:
            continue
        for i, r in enumerate(g):
            a = ans["q%d" % i]
            if a["type"] == "noul":
                p = [1 - a["noul"], a["noul"]]
            else:
                p = list(a["probabilities"].values())
            out.append({"p": p, "target": r["target"], "qtype": QTYPES[r["q"]["type"]], "task": r["task"],
                        "domain": r["domain"], "gid": r["gid"]})
    return out


def text_state(r):
    s = r.get("state") or {}
    return s if s else ""


class Florence:
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoProcessor
        try:
            from transformers import Florence2ForConditionalGeneration
            repo = "florence-community/Florence-2-large"
            self.model = Florence2ForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.float16).cuda().eval()
            self.proc = AutoProcessor.from_pretrained(repo)
        except Exception:  # noqa: BLE001 - older transformers / remote-code checkpoint
            repo = "microsoft/Florence-2-large"
            self.model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                                              trust_remote_code=True).cuda().eval()
            self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

    @torch.no_grad()
    def run(self, img, task):
        inp = self.proc(text=task, images=img, return_tensors="pt").to("cuda", torch.float16)
        ids = self.model.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"], max_new_tokens=384,
                                  num_beams=3)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def ocr_states(groups, image_root, cache_path):
    from laya.vision import load_image
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    todo = [g for g in groups if g[0]["gid"] not in cache and g[0].get("image")]
    if todo:
        fl = Florence()
        for i, g in enumerate(todo):
            img = load_image(os.path.join(image_root, g[0]["image"]))
            cache[g[0]["gid"]] = {"image_caption": fl.run(img, "<MORE_DETAILED_CAPTION>"),
                                  "ocr_text": fl.run(img, "<OCR>")[:3000]}
            if i % 200 == 0:
                print("florence", i, len(todo), flush=True)
                json.dump(cache, open(cache_path, "w"))
        json.dump(cache, open(cache_path, "w"))
        del fl
        torch.cuda.empty_cache()
    return cache


def vlm_rows(groups, image_root, model=None, n_per_task=None, include_teacher_sources=False, cache_path=None,
             concurrency=32, max_usd=None, stats=None):
    """Answers of an OpenRouter VLM on eval images, with the teacher's labelling prompt and validation.

    Groups are sent in chunks of <= 10 questions (one image per call). A chunk that still fails validation
    after one retry yields `failed=True` for its invalid questions only (uniform probabilities, scored as
    wrong), so every
    system is scored on the same rows. Raw answers are cached in `cache_path` (JSONL) so a rerun
    resumes without paying twice. `n_per_task` subsamples rows per task (the teacher column);
    teacher-labelled images are skipped unless `include_teacher_sources` is set.
    """
    import asyncio

    import teacher_label as TL
    from openrouter import TEACHER, Client, image_part, message_json
    from laya.common import QTYPES

    model = model or TEACHER
    by_task = defaultdict(list)
    for g in groups:
        by_task[g[0]["task"]].append(g)
    chunks = []
    for task, gs in by_task.items():
        n = 0
        for g in gs:
            if n_per_task is not None and n >= n_per_task:
                break
            if not g[0].get("image") or (g[0]["source"].startswith("teacher") and not include_teacher_sources):
                continue
            for k in range(0, len(g), 10):
                chunks.append(("%s#%d" % (g[0]["gid"], k), g[k:k + 10]))
            n += len(g)

    cache = {}
    if cache_path and os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            c = json.loads(line)
            cache[c["key"]] = c

    def questions(chunk):
        qs = []
        for i, r in enumerate(chunk):
            q = {"qid": "q%d" % (i + 1), "type": r["q"]["type"], "instructions": r["q"]["instructions"]}
            if "criteria" in r["q"]:
                q["criteria"] = r["q"]["criteria"]
            qs.append(q)
        return qs, {q["qid"]: list(range(len(TL.option_texts(q)))) for q in qs}

    def check(parsed, qs, orders):
        """{qid: probs} for every question with a valid answer; a bad answer only fails its own question."""
        res = TL.validate(parsed, qs, orders, allow_onehot=True) if parsed is not None else None
        if isinstance(res, dict):
            return res
        good = {}
        answers = parsed.get("answers") if isinstance(parsed, dict) else None
        for a in answers if isinstance(answers, list) else []:
            q = next((q for q in qs if isinstance(a, dict) and q["qid"] == a.get("qid")), None)
            if q is not None:
                one = TL.validate({"answers": [a]}, [q], {q["qid"]: orders[q["qid"]]}, allow_onehot=True)
                if isinstance(one, dict):
                    good.update(one)
        return good

    def to_rows(chunk, res):
        qs, _ = questions(chunk)
        out = []
        for q, r in zip(qs, chunk):
            k = len(r["target"])
            failed = q["qid"] not in res
            p = [1.0 / k] * k if failed else res[q["qid"]]
            out.append({"p": p, "target": r["target"], "qtype": QTYPES[r["q"]["type"]], "task": r["task"],
                        "domain": r["domain"], "gid": r["gid"], "qkey": r.get("qkey"), "failed": failed})
        return out

    async def go():
        client = Client(concurrency=concurrency, model=model)
        out = []
        log = open(cache_path, "a", encoding="utf-8") if cache_path else None

        async def one(key, chunk):
            qs, orders = questions(chunk)
            if key in cache:
                out.extend(to_rows(chunk, check(cache[key]["parsed"], qs, orders)))
                return
            if max_usd is not None and client.stats["usd"] >= max_usd:
                return  # budget stop: left uncached, so a rerun picks it up
            msgs = [{"role": "system", "content": TL.SYSTEM},
                    {"role": "user", "content": [{"type": "text", "text": TL.render_prompt(qs, orders)},
                                                 image_part(os.path.join(image_root, chunk[0]["image"]))]}]
            best, usage = (None, {}), {}
            for _ in range(2):
                data = await client.chat(msgs, schema=TL.SCHEMA)
                usage = (data or {}).get("usage") or usage
                parsed = message_json(data) if data else None
                res = check(parsed, qs, orders)
                if best[0] is None or len(res) > len(best[1]):
                    best = (parsed, res)
                if len(res) == len(qs):
                    break
            parsed, res = best
            if log:
                log.write(json.dumps({"key": key, "model": model, "parsed": parsed, "valid": len(res),
                                      "questions": len(qs), "usage": usage}) + "\n")
                log.flush()
            out.extend(to_rows(chunk, res))

        await asyncio.gather(*(one(k, c) for k, c in chunks))
        await client.close()
        if log:
            log.close()
        return out, client.stats

    rows, st = asyncio.run(go())
    if stats is not None:
        stats.update({"calls": len(chunks), "usd": st["usd"], "ok": st["ok"], "http_err": st["http_err"],
                      "schema_fallback": st["schema_fallback"], "latency_s": st["latency"],
                      "failed_rows": sum(r["failed"] for r in rows), "rows": len(rows)})
    return rows


def select_groups(items, per_task, seed=5):
    """The fixed eval subset: <= per_task rows per task, whole image groups, shuffled with `seed`."""
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for g in vdata.group_items([r for r in items if r.get("image")]):
        by_task[g[0]["task"]].append(g)
    groups = []
    for task, gs in sorted(by_task.items()):
        rng.shuffle(gs)
        n = 0
        for g in gs:
            if n >= per_task:
                break
            groups.append(g)
            n += len(g)
    return groups, rng


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--stage-a")
    ap.add_argument("--base", required=True)
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=600)
    ap.add_argument("--teacher-per-task", type=int, default=150)
    ap.add_argument("--audit", help="audit verdicts JSON exported from audit.html")
    ap.add_argument("--audit-items", help="audit_items.jsonl the verdicts refer to")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--head-max-len", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    groups, rng = select_groups(vdata.read_items(args.items, "eval"), args.per_task)
    print("eval groups", len(groups), "rows", sum(len(g) for g in groups), flush=True)

    report = {}

    def record(name, rows):
        report[name] = metrics(rows)
        json.dump(report, open(os.path.join(args.out, "eval_report.json"), "w"), indent=1)
        print(name, json.dumps({k: round(v["acc"], 4) for k, v in report[name].items()
                                if k == "all" or k.startswith("task:")}), flush=True)

    final_rows = laya_v_rows(args.final, groups, args)
    record("laya_v", final_rows)
    if args.stage_a:
        record("laya_v_A", laya_v_rows(args.stage_a, groups, args))

    from laya.agent import Agent
    base = Agent(args.base, device="cuda")
    if "text_only" not in args.skip:
        record("text_only", agent_rows(base, groups, lambda g: text_state(g[0])))
    if "ocr" not in args.skip:
        cache = ocr_states(groups, args.image_root, os.path.join(args.out, "florence_cache.json"))

        def ocr_state(g):
            s = text_state(g[0])
            s = dict(s) if isinstance(s, dict) else ({"text": s} if s else {})
            s.update(cache.get(g[0]["gid"], {}))
            return s
        record("ocr_pipeline", agent_rows(base, groups, ocr_state))
    del base
    torch.cuda.empty_cache()
    if "teacher" not in args.skip and os.environ.get("OPENROUTER_API_KEY"):
        record("teacher", vlm_rows(groups, args.image_root, n_per_task=args.teacher_per_task))

    # ---- failure probes
    if "probes" not in args.skip:
        from PIL import Image
        gids = [g[0]["gid"] for g in groups]
        perm = gids[1:] + gids[:1]
        path_of = {g[0]["gid"]: g[0]["image"] for g in groups}
        swap = {a: path_of[b] for a, b in zip(gids, perm)}
        from laya.vision import load_image

        def shuffled(gid, img):
            return load_image(os.path.join(args.image_root, swap[gid]))
        record("probe_shuffled_images", laya_v_rows(args.final, groups, args, image_override=shuffled))

        def grey(gid, img):
            return Image.new("RGB", img.size, (128, 128, 128))
        grey_rows = laya_v_rows(args.final, groups, args, image_override=grey)
        record("probe_blank_image", grey_rows)
        report["probe_blank_image"]["mean_max_prob"] = float(np.mean([max(r["p"]) for r in grey_rows]))
        report["laya_v"]["all"]["mean_max_prob"] = float(np.mean([max(r["p"]) for r in final_rows]))

        caps = {}
        cap_file = os.path.join(args.image_root, "items", "captions_coco.jsonl")
        if os.path.exists(cap_file):
            for line in open(cap_file, encoding="utf-8"):
                c = json.loads(line)
                caps[c["gid"]] = c["captions"][0]
        photo = [g for g in groups if g[0]["domain"] == "photo"]
        other = list(caps.values())

        def contradict(r):
            return {"caption": rng.choice(other)} if other else {"caption": "a photo of a cat"}
        rows = laya_v_rows(args.final, photo, args, state_override=contradict)
        record("probe_contradicting_caption", rows)
        record("laya_v_photo_only", [r for r in final_rows if r["domain"] == "photo"])

    # ---- audit: human verdicts exported by audit.html, aligned by index with audit_items.jsonl
    if args.audit and os.path.exists(args.audit):
        items = vdata.read_items([args.audit_items])
        raw = json.load(open(args.audit, encoding="utf-8"))
        verdicts = {v["i"]: v.get("human") for v in raw}
        judge = next((v["judge"] for v in raw if v.get("judge")), "human")
        judged = [(i, it) for i, it in enumerate(items) if verdicts.get(i) is not None]
        rows = laya_v_rows(args.final, [[dict(it, gid="audit-%d" % i)] for i, it in judged], args)
        pred = {r["gid"]: int(np.argmax(r["p"])) for r in rows}
        t_ok = [float(int(np.argmax(it["target"])) == verdicts[i]) for i, it in judged]
        l_ok = [float(pred.get("audit-%d" % i, -1) == verdicts[i]) for i, it in judged]
        by_dom = defaultdict(lambda: [[], []])
        for (i, it), a1, a2 in zip(judged, t_ok, l_ok):
            by_dom[it["domain"]][0].append(a1)
            by_dom[it["domain"]][1].append(a2)
        report["audit"] = {"n": len(judged), "reference": judge, "teacher_acc": float(np.mean(t_ok)), "laya_v_acc": float(np.mean(l_ok)),
                           "laya_v_over_teacher": float(np.mean(l_ok) / max(1e-9, np.mean(t_ok))),
                           "by_domain": {d: {"n": len(v[0]), "teacher": float(np.mean(v[0])),
                                             "laya_v": float(np.mean(v[1]))} for d, v in by_dom.items()}}
        print("audit", json.dumps(report["audit"]), flush=True)
    json.dump(report, open(os.path.join(args.out, "eval_report.json"), "w"), indent=1)
    print("wrote", os.path.join(args.out, "eval_report.json"))


if __name__ == "__main__":
    main()
