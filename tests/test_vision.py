"""Laya-V image input: sequence layout, splice, preprocessing, and the Agent path end to end.

Runs offline on CPU with a tiny from-config checkpoint (ModernBERT text encoder, SigLIP 2
vision tower, word-level tokenizer) written to a temp dir and loaded through `Agent`, so the
real load / encode / collate / forward / decode path is exercised.

    python tests/test_vision.py        (or: python -m pytest tests/test_vision.py)
"""
import base64
import copy
import io
import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from laya.agent import Agent  # noqa: E402
from laya.common import DecisionModel, build_sequence, collate_items, temp_key  # noqa: E402
from laya.vision import (IMAGE_SLOTS, VisionAdapter, choose_budget, load_image, preprocess,  # noqa: E402
                         vision_config_for)

WORDS = ("the a is of and to in receipt total over limit food travel other yes no image photo "
         "cat dog what was bought question choice score noul level statement holds does not "
         "note claim reimbursement meals transport hotels everything else how urgent").split()


# ------------------------------------------------------------------ tiny checkpoint
def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["[PAD]", "[MASK]", "[CLS]", "[SEP]", "[UNK]"]
    vocab = {t: i for i, t in enumerate(specials + WORDS + list("0123456789:.,{}\"_-?"))}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits(True)])
    return PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", mask_token="[MASK]",
                                   cls_token="[CLS]", sep_token="[SEP]", unk_token="[UNK]")


def _tiny_parts(seed=0):
    from transformers import AutoModel, ModernBertConfig, Siglip2VisionConfig, Siglip2VisionModel

    torch.manual_seed(seed)
    tok = _tokenizer()
    ecfg = ModernBertConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                            num_attention_heads=2, global_attn_every_n_layers=1, local_attention=16,
                            max_position_embeddings=1024, pad_token_id=tok.pad_token_id,
                            cls_token_id=tok.cls_token_id, sep_token_id=tok.sep_token_id,
                            bos_token_id=tok.cls_token_id, eos_token_id=tok.sep_token_id)
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    vcfg_t = Siglip2VisionConfig(hidden_size=24, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2,
                                 patch_size=16, num_patches=64)
    tower = Siglip2VisionModel(vcfg_t)
    tower_cfg = {k: v for k, v in vcfg_t.to_dict().items()
                 if k not in ("architectures", "transformers_version", "torch_dtype", "dtype")}
    vcfg = vision_config_for("tiny/siglip2", tower_cfg, resampler_heads=2)
    adapter = VisionAdapter(vcfg, 32, tower=tower)
    model = DecisionModel(enc, head_layers=1, n_act=2, vision=adapter).eval()
    return tok, ecfg, vcfg, model


def make_checkpoint(path, vision=True, seed=0):
    from safetensors.torch import save_file

    tok, ecfg, vcfg, model = _tiny_parts(seed)
    os.makedirs(path, exist_ok=True)
    tok.save_pretrained(os.path.join(path, "tokenizer"))
    ecfg.save_pretrained(os.path.join(path, "encoder"))
    cfg = {"encoder": "tiny/modernbert", "head_layers": 1, "max_len": 512, "head_max_len": 128,
           "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {}}
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    if vision:
        cfg["vision"] = vcfg
    else:
        sd = {k: v for k, v in sd.items() if not k.startswith("vision.")}
    with open(os.path.join(path, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f)
    save_file(sd, os.path.join(path, "model.safetensors"))
    return path


_TMP = tempfile.mkdtemp(prefix="laya_v_test_")
VISION_DIR = make_checkpoint(os.path.join(_TMP, "vision"), vision=True)
TEXT_DIR = make_checkpoint(os.path.join(_TMP, "text"), vision=False)

QUESTIONS = {
    "over_limit": {"type": "noul", "instructions": "is the receipt total over limit ?"},
    "category": {"type": "choice", "instructions": "what was bought ?",
                 "criteria": {"meals": "food", "travel": "transport , hotels", "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "how urgent ?", "criteria": ["no", "a", "the"]},
}


def _img(seed=0, size=(96, 64)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8))


class FakeTok:
    mask_token, mask_token_id, cls_token_id, sep_token_id, pad_token_id = "[MASK]", 1, 2, 3, 0

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        n = max(1, len(text) // 4)
        if truncation and max_length:
            n = min(n, max_length)
        return {"input_ids": [5] * n}


Q = {"t": "choice", "ins": "pick one", "crit": {"a": "x", "b": "y"}}


# ------------------------------------------------------------------ build_sequence
def test_text_only_sequence_unchanged():
    ids, markers = build_sequence(FakeTok(), "some state text " * 10, Q, 128, 64)
    ids0, markers0 = build_sequence(FakeTok(), "some state text " * 10, Q, 128, 64, image_slots=0)
    assert (ids, markers) == (ids0, markers0)


def test_image_slots_keep_markers_and_reserve_state_start():
    tok = FakeTok()
    state = "x" * 4000
    ids_t, mk_t = build_sequence(tok, state, Q, 512, 64)
    ids_i, mk_i, pos = build_sequence(tok, state, Q, 512, 64, image_slots=IMAGE_SLOTS)
    assert mk_i == mk_t
    assert ids_i[pos - 1] == tok.sep_token_id  # right after the option block's [SEP]
    assert ids_i[:pos] == ids_t[:pos]
    assert ids_i[pos:pos + IMAGE_SLOTS] == [tok.pad_token_id] * IMAGE_SLOTS
    assert len(ids_i) == len(ids_t) == 512  # the state gives up exactly the image's slots
    assert ids_i[-1] == tok.sep_token_id


def test_image_slots_too_small_max_len_raises():
    try:
        build_sequence(FakeTok(), "s", Q, 40, 32, image_slots=IMAGE_SLOTS)
    except ValueError as e:
        assert "image slots" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ------------------------------------------------------------------ model splice
def test_inputs_embeds_path_matches_input_ids():
    _, _, _, model = _tiny_parts()
    ids = torch.randint(5, 40, (3, 20))
    att = torch.ones_like(ids)
    att[1, 15:] = 0
    with torch.no_grad():
        a = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        b = model.encoder(inputs_embeds=model.encoder.get_input_embeddings()(ids), attention_mask=att).last_hidden_state
    assert torch.allclose(a[att.bool()], b[att.bool()], atol=1e-5)


def test_splice_writes_exactly_the_reserved_slots():
    _, _, _, model = _tiny_parts()
    ids = torch.randint(5, 40, (3, 200))
    vis = torch.randn(2, IMAGE_SLOTS, 32)
    vis_index = torch.tensor([1, -1, 0])
    img_pos = torch.tensor([10, -1, 7])
    emb = model.embed_with_images(ids, vis, vis_index, img_pos)
    ref = model.encoder.get_input_embeddings()(ids)
    assert torch.equal(emb[1], ref[1])
    assert torch.equal(emb[0, 10:10 + IMAGE_SLOTS], vis[1])
    assert torch.equal(emb[2, 7:7 + IMAGE_SLOTS], vis[0])
    assert torch.equal(emb[0, :10], ref[0, :10]) and torch.equal(emb[0, 10 + IMAGE_SLOTS:], ref[0, 10 + IMAGE_SLOTS:])


def test_splice_is_differentiable_into_adapter():
    _, _, _, model = _tiny_parts()
    model.train()
    batch = preprocess([_img(1)], [64], 16)
    vis = model.vision(**batch)
    ids = torch.randint(5, 40, (1, 160))
    att = torch.ones_like(ids)
    logits, _ = model(ids, att, torch.tensor([[1, 2]]), torch.ones(1, 2, dtype=torch.bool), torch.tensor([0]),
                      vis_tokens=vis, vis_index=torch.tensor([0]), img_pos=torch.tensor([5]))
    logits.sum().backward()
    assert any(p.grad is not None for p in model.vision.resampler.parameters())
    assert model.vision.img_start.grad is not None
    assert all(p.grad is None for p in model.vision.tower.parameters())


# ------------------------------------------------------------------ preprocessing
def test_preprocess_matches_siglip2_processor():
    from transformers.models.siglip2.image_processing_pil_siglip2 import Siglip2ImageProcessorPil

    img = _img(3, size=(300, 170))
    ref = Siglip2ImageProcessorPil().preprocess([img], max_num_patches=256, patch_size=16, return_tensors="pt")
    ours = preprocess([img], [256], 16)
    assert torch.equal(ours["spatial_shapes"], ref["spatial_shapes"].long())
    assert torch.equal(ours["pixel_attention_mask"], ref["pixel_attention_mask"].long())
    assert torch.allclose(ours["pixel_values"], ref["pixel_values"].float(), atol=2e-2)


def test_preprocess_pads_mixed_budgets():
    out = preprocess([_img(1, (64, 64)), _img(2, (400, 100))], [16, 64], 16)
    assert out["pixel_values"].shape[:2] == (2, 64)
    assert int(out["pixel_attention_mask"][0].sum()) <= 16
    assert int(out["pixel_attention_mask"][1].sum()) <= 64


def test_load_image_forms_agree():
    img = _img(4)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    raw = buf.getvalue()
    path = os.path.join(_TMP, "x.png")
    with open(path, "wb") as f:
        f.write(raw)
    b64 = base64.b64encode(raw).decode()
    forms = [img, raw, path, b64, "data:image/png;base64," + b64]
    arrs = [np.asarray(load_image(x)) for x in forms]
    assert all(np.array_equal(arrs[0], a) for a in arrs[1:])


def test_choose_budget():
    assert choose_budget(_img(0, (640, 480))) == 576
    assert choose_budget(_img(0, (400, 1200))) == 1024
    assert choose_budget(_img(0, (640, 480)), "low") == 256


# ------------------------------------------------------------------ Agent end to end
def _probs(ans):
    out = []
    for k in sorted(ans):
        a = ans[k]
        out.append(a["noul"] if a["type"] == "noul" else list(a["probabilities"].values()))
    return out


def test_vision_checkpoint_text_parity():
    """Text-only requests on the vision checkpoint answer exactly as the text-only checkpoint."""
    va, ta = Agent(VISION_DIR, device="cpu"), Agent(TEXT_DIR, device="cpu")
    for state in ["a receipt total over limit", {"note": "food and travel"}, [{"role": "user", "content": "yes"}]]:
        assert va.predict(state, QUESTIONS)["answers"] == ta.predict(state, QUESTIONS)["answers"]


def test_image_predict_all_types_and_input_forms():
    agent = Agent(VISION_DIR, device="cpu")
    img = _img(5)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    r1 = agent.predict({"image": img, "note": "reimbursement claim"}, QUESTIONS)
    r2 = agent.predict({"image": buf.getvalue(), "note": "reimbursement claim"}, QUESTIONS)
    assert set(r1["answers"]) == set(QUESTIONS)
    assert r1["answers"]["category"]["choice"] in ("meals", "travel", "other")
    assert 0.0 <= r1["answers"]["over_limit"]["noul"] <= 1.0
    np.testing.assert_allclose(np.array(_probs(r1["answers"])[0]), np.array(_probs(r2["answers"])[0]), atol=1e-6)


def test_image_changes_the_logits():
    # Raw logits, not answers: an untrained tiny head scores every option of a row alike, so
    # its probabilities are uniform whatever the input -- but the image must reach the markers.
    agent = Agent(VISION_DIR, device="cpu")
    internal = {k: agent._to_internal(v) for k, v in QUESTIONS.items()}
    outs = []
    for img in (_img(6), _img(7, (50, 200))):
        items = agent._encode_state({"image": img}, list(internal), internal)
        for it in items:
            it["image_idx"] = 0
        b = collate_items([items], agent.tok.pad_token_id)
        b["images"] = [items[0]["image"]]
        with torch.no_grad():
            outs.append(agent._infer(b)[0])
    assert (outs[0] - outs[1]).abs().max() > 1e-4


def test_mixed_batch_matches_single_predictions():
    agent = Agent(VISION_DIR, device="cpu")
    states = [{"image": _img(8), "note": "food"}, "travel total over limit", {"image": _img(9, (200, 80))}]
    batch = agent.predict_batch(states, QUESTIONS)
    for st, res in zip(states, batch):
        single = agent.predict(st, QUESTIONS)
        for x, y in zip(_probs(res["answers"]), _probs(single["answers"])):
            np.testing.assert_allclose(np.array(x), np.array(y), atol=1e-4)


def test_image_on_text_checkpoint_raises():
    agent = Agent(TEXT_DIR, device="cpu")
    try:
        agent.predict({"image": _img(1)}, QUESTIONS)
    except ValueError as e:
        assert "vision" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_image_temperatures_apply_only_to_image_rows():
    agent = Agent(VISION_DIR, device="cpu")
    base_img = agent.predict({"image": _img(2)}, QUESTIONS)["answers"]
    base_txt = agent.predict("food", QUESTIONS)["answers"]
    agent.temperature_by_options = dict(agent.temperature_by_options)
    agent.temperature_by_options[temp_key(0, 3, "image")] = 3.0
    hot_img = agent.predict({"image": _img(2)}, QUESTIONS)["answers"]
    assert agent.predict("food", QUESTIONS)["answers"] == base_txt
    p0 = np.array(list(base_img["category"]["probabilities"].values()))
    p1 = np.array(list(hot_img["category"]["probabilities"].values()))
    assert p1.max() - p1.min() <= p0.max() - p0.min() + 1e-9  # softened
    assert hot_img["over_limit"] == base_img["over_limit"]


def test_predict_long_rejects_images():
    agent = Agent(VISION_DIR, device="cpu")
    try:
        agent.predict_long({"image": _img(1)}, QUESTIONS)
    except ValueError as e:
        assert "image" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_collate_mixed_rows():
    items_img = [{"ids": [1, 2, 3], "markers": [1], "qtype": 0, "img_pos": 2, "image_idx": 0, "image": object()}]
    items_txt = [{"ids": [1, 2], "markers": [1], "qtype": 0}]
    b = collate_items([items_img, items_txt], pad_id=0)
    assert b["img_pos"].tolist() == [2, -1] and b["vis_index"].tolist() == [0, -1]
    assert "image" not in b["meta"][0]
    assert "img_pos" not in collate_items([copy.deepcopy(items_txt)], pad_id=0)


# ------------------------------------------------------------------ Router
def test_router_routes_images_to_vision():
    from laya.router import Router

    r = Router()
    d = r.route({"image": _img(1), "note": "Reimbursement claim"}, QUESTIONS)
    assert d["model"] == "vision" and d["reason"] == "state contains an image"
    assert r.route({"note": "Reimbursement claim"}, QUESTIONS)["model"] == "english"
    assert r.route({"image": _img(1)}, QUESTIONS, model="english")["model"] == "english"  # explicit wins
    assert r.route({"image": None, "m": "hola"}, QUESTIONS)["model"] != "vision"


def test_router_predicts_through_vision_checkpoint():
    from laya.router import Router

    r = Router(models={"vision": VISION_DIR, "english": TEXT_DIR}, device="cpu")
    out = r.predict({"image": _img(2), "note": "food"}, QUESTIONS)
    assert out["routing"]["model"] == "vision"
    assert set(out["answers"]) == set(QUESTIONS)
    assert r.loaded == ["vision"]


def test_router_default_preload_skips_vision():
    from unittest.mock import patch

    from laya.router import Router

    with patch("laya.agent.Agent", side_effect=lambda repo, **kw: object()):
        r = Router(preload=True)
        assert "vision" not in r.loaded and len(r.loaded) == 3


# ------------------------------------------------------------------ network image input
def test_resolve_image_field_base64_and_refusals():
    from laya.vision import resolve_image_field

    buf = io.BytesIO()
    _img(1).save(buf, "PNG")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    assert resolve_image_field(b64) == raw
    assert resolve_image_field("data:image/png;base64," + b64) == raw
    for bad, allow in [("http://127.0.0.1/x.png", True), ("http://10.0.0.5/x.png", True),
                       ("file:///etc/passwd", True), ("https://example.com/x.png", False), ("not base64!", True),
                       (123, True)]:
        try:
            resolve_image_field(bad, allow_urls=allow)
        except ValueError:
            pass
        else:
            raise AssertionError("expected refusal for %r" % (bad,))


def test_serve_accepts_base64_image_state():
    from fastapi.testclient import TestClient

    from laya.router import Router
    from laya.serve import create_app

    router = Router(models={"vision": VISION_DIR, "english": TEXT_DIR}, device="cpu")
    buf = io.BytesIO()
    _img(3).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    with TestClient(create_app(router)) as c:
        r = c.post("/v1/systemone", json={"state": {"image": b64, "note": "food"}, "questions": QUESTIONS})
        assert r.status_code == 200, r.text
        assert r.json()["routing"]["model"] == "vision"
        r = c.post("/v1/systemone", json={"state": {"image": "http://127.0.0.1/a.png"}, "questions": QUESTIONS})
        assert r.status_code == 422
        big = {"state": {"note": "x" * 10}, "questions": QUESTIONS, "pad": "y" * (3 * 1024 * 1024)}
        assert c.post("/v1/systemone", json=big).status_code == 413


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if fails else 0)
