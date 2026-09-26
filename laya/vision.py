"""Laya-V: image input for Laya decision models.

A frozen SigLIP 2 NaFlex vision tower reads the image at its native aspect ratio; a Perceiver
resampler turns its variable-length patch sequence into a fixed number of tokens; a small MLP
projects them to the text encoder's width. The tokens, framed by two learned boundary vectors,
replace reserved placeholder slots at the start of each question row's state segment (see
`build_sequence(image_slots=...)` and `DecisionModel.forward(vis_tokens=...)`), so the option
markers and the decision head are untouched.

Everything here is optional: a checkpoint without a `vision` config never imports this module.
"""
import base64
import io
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

IMAGE_TOKENS = 128
IMAGE_SLOTS = IMAGE_TOKENS + 2  # plus the <img> and </img> boundary vectors
PATCH_BUDGETS = {"low": 256, "mid": 576, "high": 1024}
# Refuse decompression bombs before PIL allocates them.
MAX_IMAGE_PIXELS = 64_000_000
MAX_IMAGE_BYTES = 32 * 1024 * 1024


# --------------------------------------------------------------------------- image input
def is_image_state(state: Any) -> bool:
    return isinstance(state, dict) and state.get("image") is not None


def load_image(x: Any):
    """A PIL RGB image from a PIL image, a file path, raw bytes, or a base64 / data-URI string."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    if isinstance(x, Image.Image):
        img = x
    elif isinstance(x, (bytes, bytearray, memoryview)):
        if len(x) > MAX_IMAGE_BYTES:
            raise ValueError("image is %d bytes; the limit is %d" % (len(x), MAX_IMAGE_BYTES))
        img = Image.open(io.BytesIO(bytes(x)))
    elif isinstance(x, (str, os.PathLike)):
        s = os.fspath(x)
        if isinstance(x, str) and (s.startswith("data:") or (len(s) > 512 and not os.path.exists(s))):
            payload = s.split(",", 1)[1] if s.startswith("data:") else s
            try:
                raw = base64.b64decode(payload, validate=False)
            except (ValueError, TypeError) as e:
                raise ValueError("image string is neither a file path nor valid base64") from e
            return load_image(raw)
        if not os.path.exists(s):
            raise FileNotFoundError("image file not found: %r" % s)
        img = Image.open(s)
    else:
        raise TypeError("state['image'] must be a PIL image, a path, bytes or base64 string, not %s"
                        % type(x).__name__)
    img.load()
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise ValueError("image has %d pixels; the limit is %d" % (img.width * img.height, MAX_IMAGE_PIXELS))
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            img = bg
        else:
            img = img.convert("RGB")
    return img


def _public_host(host: str) -> bool:
    """True only if every address `host` resolves to is a public unicast address."""
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
                or ip.is_unspecified):
            return False
    return True


def fetch_image_url(url: str, max_bytes: int = MAX_IMAGE_BYTES, timeout: float = 10.0) -> bytes:
    """Download an image for a server request: http(s) only, public hosts only, no redirects,
    capped size and time. Raises ValueError with a client-safe message on any refusal."""
    import urllib.parse
    import urllib.request

    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("image URL must be http(s)")
    if not _public_host(parts.hostname):
        raise ValueError("image URL host is not a public address")

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(urllib.request.Request(url, headers={"User-Agent": "laya-v"}), timeout=timeout) as r:
            if r.status != 200:
                raise ValueError("image URL returned HTTP %d" % r.status)
            data = r.read(max_bytes + 1)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001 - network errors become a client error, not a 500
        raise ValueError("could not fetch image URL (%s)" % type(e).__name__) from e
    if len(data) > max_bytes:
        raise ValueError("image at URL exceeds %d bytes" % max_bytes)
    return data


def resolve_image_field(value: Any, allow_urls: bool = True, max_bytes: int = MAX_IMAGE_BYTES,
                        timeout: float = 10.0) -> bytes:
    """Turn a JSON `state.image` value (base64, data URI, or http(s) URL) into image bytes.

    For network surfaces (HTTP server, MCP), which must never read local file paths."""
    if not isinstance(value, str) or not value:
        raise ValueError("state.image must be a base64 string, a data: URI or an http(s) URL")
    if value.startswith(("http://", "https://")):
        if not allow_urls:
            raise ValueError("image URLs are disabled on this server; send base64")
        return fetch_image_url(value, max_bytes, timeout)
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    if len(payload) > (max_bytes * 4) // 3 + 4:
        raise ValueError("image exceeds %d bytes" % max_bytes)
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as e:
        raise ValueError("state.image is not valid base64") from e


def choose_budget(img, detail: Optional[str] = None) -> int:
    """Patch budget for one image: explicit `image_detail`, else a size/aspect heuristic.

    Training uses fixed per-domain budgets (photos 256, charts/screens 576, documents 1024);
    at inference the domain is unknown, so large or extreme-aspect images (documents, receipts,
    long screenshots) get 1024 and the rest 576.
    """
    if detail is not None:
        if isinstance(detail, int):
            return int(detail)
        if detail not in PATCH_BUDGETS:
            raise ValueError("image_detail must be one of %s or an int" % sorted(PATCH_BUDGETS))
        return PATCH_BUDGETS[detail]
    w, h = img.size
    aspect = max(w, h) / max(1, min(w, h))
    if aspect >= 1.8 or w * h >= 1600 * 1200:
        return PATCH_BUDGETS["high"]
    return PATCH_BUDGETS["mid"]


def _target_size(h: int, w: int, patch: int, max_patches: int) -> Tuple[int, int]:
    """Largest (h, w), multiples of `patch`, with at most `max_patches` patches (SigLIP 2 NaFlex rule)."""
    def scaled(scale, size):
        return max(patch, int(math.ceil(size * scale / patch) * patch))

    lo, hi = 1e-6, 100.0
    while hi - lo >= 1e-5:
        mid = (lo + hi) / 2
        if (scaled(mid, h) / patch) * (scaled(mid, w) / patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    return scaled(lo, h), scaled(lo, w)


def patchify(img, budget: int, patch_size: int = 16) -> Tuple[np.ndarray, Tuple[int, int]]:
    """One image to NaFlex patches, identical to Siglip2ImageProcessor: resize to fit the patch
    budget keeping aspect ratio, scale to [-1, 1], cut into (gh * gw, patch * patch * 3) patches."""
    from PIL import Image

    th, tw = _target_size(img.height, img.width, patch_size, int(budget))
    arr = np.asarray(img.resize((tw, th), Image.BILINEAR), dtype=np.float32)
    arr = (arr / 255.0 - 0.5) / 0.5  # H, W, C
    gh, gw = th // patch_size, tw // patch_size
    p = arr.reshape(gh, patch_size, gw, patch_size, 3).transpose(0, 2, 1, 3, 4)
    return p.reshape(gh * gw, patch_size * patch_size * 3), (gh, gw)


def pad_patches(seqs: Sequence[np.ndarray], shapes: Sequence[Tuple[int, int]],
                length: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Stack per-image patch sequences, zero-padded to `length` (default: the longest)."""
    n = int(length or max(len(s) for s in seqs))
    dim = seqs[0].shape[1]
    pv = np.zeros((len(seqs), n, dim), dtype=np.float32)
    mask = np.zeros((len(seqs), n), dtype=np.int64)
    for i, s in enumerate(seqs):
        pv[i, : len(s)] = s
        mask[i, : len(s)] = 1
    return {"pixel_values": torch.from_numpy(pv), "pixel_attention_mask": torch.from_numpy(mask),
            "spatial_shapes": torch.tensor(list(shapes), dtype=torch.long)}


def preprocess(images: Sequence[Any], budgets: Sequence[int], patch_size: int = 16) -> Dict[str, torch.Tensor]:
    """NaFlex preprocessing for a batch: `patchify` each image, pad to the largest budget."""
    out = [patchify(img, b, patch_size) for img, b in zip(images, budgets)]
    return pad_patches([o[0] for o in out], [o[1] for o in out], max(int(b) for b in budgets))


# --------------------------------------------------------------------------- modules
class ResamplerLayer(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_s = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_f = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), nn.GELU(), nn.Linear(dim * ff_mult, dim))

    def forward(self, q, kv, kv_pad):
        kv = self.ln_kv(kv)
        x = self.ln_q(q)
        q = q + self.cross(x, kv, kv, key_padding_mask=kv_pad, need_weights=False)[0]
        x = self.ln_s(q)
        q = q + self.self_attn(x, x, x, need_weights=False)[0]
        return q + self.ff(self.ln_f(q))


class Resampler(nn.Module):
    """Perceiver resampler: `n_queries` learned queries cross-attend to a variable patch sequence."""

    def __init__(self, in_dim: int, dim: int, n_queries: int = IMAGE_TOKENS, layers: int = 2,
                 heads: int = 12, ff_mult: int = 2, query_init_std: float = 0.02):
        super().__init__()
        self.in_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * query_init_std)
        self.layers = nn.ModuleList([ResamplerLayer(dim, heads, ff_mult) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, feat_mask):
        kv = self.in_proj(feats)
        pad = ~feat_mask.bool()
        q = self.queries.unsqueeze(0).expand(feats.size(0), -1, -1).to(kv.dtype)
        for layer in self.layers:
            q = layer(q, kv, pad)
        return self.norm(q)


GRID_SHAPES = [(h, IMAGE_TOKENS // h) for h in (1, 2, 4, 8, 16, 32, 64, 128) if IMAGE_TOKENS % h == 0]


def grid_shape(gh: int, gw: int, n_tokens: int = IMAGE_TOKENS):
    """Output grid (h, w) with h * w == n_tokens whose aspect ratio is closest to the patch grid's."""
    import math
    shapes = [(h, n_tokens // h) for h in range(1, n_tokens + 1) if n_tokens % h == 0]
    target = math.log(max(1, gw) / max(1, gh))
    return min(shapes, key=lambda hw: abs(math.log(hw[1] / hw[0]) - target))


class GridPooler(nn.Module):
    """Average-pools each image's NaFlex patch grid to exactly `n_tokens` tokens (raster order),
    with an output grid shaped like the image. Carries local, image-specific features from the
    first step with nothing to learn (unlike a randomly initialised resampler)."""

    def __init__(self, in_dim: int, dim: int, n_tokens: int = IMAGE_TOKENS):
        super().__init__()
        self.n_tokens = n_tokens
        self.proj = nn.Linear(in_dim, dim)

    def forward(self, feats, feat_mask, spatial_shapes):
        out = []
        for i in range(feats.size(0)):
            gh, gw = (int(x) for x in spatial_shapes[i])
            grid = feats[i, : gh * gw].reshape(gh, gw, -1).permute(2, 0, 1)[None]  # 1, C, gh, gw
            oh, ow = grid_shape(gh, gw, self.n_tokens)
            pooled = torch.nn.functional.adaptive_avg_pool2d(grid.float(), (oh, ow)).to(feats.dtype)
            out.append(pooled[0].permute(1, 2, 0).reshape(oh * ow, -1))
        return self.proj(torch.stack(out))


def _tower_from_config(tower_config: Dict, no_init: bool):
    from transformers import Siglip2VisionConfig, Siglip2VisionModel

    cfg = Siglip2VisionConfig(**{k: v for k, v in tower_config.items() if k not in ("model_type", "transformers_version")})
    if no_init:
        from .common import _no_init_weights
        with _no_init_weights():
            return Siglip2VisionModel(cfg)
    return Siglip2VisionModel(cfg)


class VisionAdapter(nn.Module):
    """Frozen SigLIP 2 tower + trainable resampler, projector and boundary vectors."""

    def __init__(self, vcfg: Dict, text_dim: int, tower: Optional[nn.Module] = None, no_init: bool = False):
        super().__init__()
        self.vcfg = dict(vcfg)
        self.tower = tower if tower is not None else _tower_from_config(vcfg["tower_config"], no_init)
        in_dim = self.tower.config.hidden_size
        rdim = int(vcfg.get("resampler_dim", text_dim))
        self.n_tokens = int(vcfg.get("image_tokens", IMAGE_TOKENS))
        self.pooler_kind = vcfg.get("pooler", "resampler")
        if self.pooler_kind == "grid":
            self.resampler = GridPooler(in_dim, rdim, self.n_tokens)
        else:
            self.resampler = Resampler(in_dim, rdim, self.n_tokens, int(vcfg.get("resampler_layers", 2)),
                                       int(vcfg.get("resampler_heads", max(1, rdim // 64))),
                                       int(vcfg.get("resampler_ff_mult", 2)),
                                       float(vcfg.get("query_init_std", 0.02)))
        self.projector = nn.Sequential(nn.Linear(rdim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim))
        self.img_start = nn.Parameter(torch.randn(text_dim) * 0.02)
        self.img_end = nn.Parameter(torch.randn(text_dim) * 0.02)
        # Per-channel standardisation of tower features, measured once on training images
        # (`fit_feature_stats`). SigLIP patch features share a large component across all
        # images (mean-pooled features of unrelated images have cosine ~0.7); left in, an
        # untrained resampler passes that component through and every image looks the same.
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.freeze_tower()

    @property
    def patch_size(self) -> int:
        return int(self.tower.config.patch_size)

    @property
    def slots(self) -> int:
        return self.n_tokens + 2

    def freeze_tower(self):
        for p in self.tower.parameters():
            p.requires_grad_(False)
        self.tower.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.tower.eval()  # frozen for the whole run, dropout never on
        return self

    def tower_features(self, pixel_values, pixel_attention_mask, spatial_shapes):
        with torch.no_grad():
            dtype = next(self.tower.parameters()).dtype
            out = self.tower(pixel_values=pixel_values.to(dtype), pixel_attention_mask=pixel_attention_mask,
                             spatial_shapes=spatial_shapes)
        return out.last_hidden_state

    def forward(self, pixel_values, pixel_attention_mask, spatial_shapes, feats: Optional[torch.Tensor] = None):
        """[n_images, n_tokens + 2, text_dim]: <img>, resampled tokens, </img>."""
        if feats is None:
            feats = self.tower_features(pixel_values, pixel_attention_mask, spatial_shapes)
        feats = (feats.float() - self.feat_mean.float()) / self.feat_std.float()
        if self.pooler_kind == "grid":
            x = self.projector(self.resampler(feats.to(self.img_start.dtype), pixel_attention_mask, spatial_shapes))
        else:
            x = self.projector(self.resampler(feats.to(self.img_start.dtype), pixel_attention_mask))
        n = x.size(0)
        start = self.img_start.to(x.dtype).expand(n, 1, -1)
        end = self.img_end.to(x.dtype).expand(n, 1, -1)
        return torch.cat([start, x, end], dim=1)


@torch.no_grad()
def fit_feature_stats(adapter: "VisionAdapter", batches, device) -> int:
    """Set `feat_mean` / `feat_std` from tower features of `batches` (dicts of NaFlex pixel tensors).
    Returns the number of patches measured."""
    total, s1, s2 = 0.0, None, None
    for px in batches:
        px = {k: v.to(device) for k, v in px.items()}
        f = adapter.tower_features(**px).float()
        m = px["pixel_attention_mask"].float()[..., None]
        a, b = (f * m).sum((0, 1)), (f * f * m).sum((0, 1))
        s1 = a if s1 is None else s1 + a
        s2 = b if s2 is None else s2 + b
        total += float(m.sum())
    mean = s1 / total
    std = (s2 / total - mean * mean).clamp_min(1e-8).sqrt()
    adapter.feat_mean.copy_(mean)
    adapter.feat_std.copy_(std + 1e-6)
    return int(total)


def vision_config_for(encoder_id: str, tower_config: Dict, **overrides) -> Dict:
    """The `vision` block stored in rl_agent_config.json."""
    cfg = {"encoder": encoder_id, "tower_config": tower_config, "image_tokens": IMAGE_TOKENS, "pooler": "grid",
           "resampler_layers": 2, "resampler_ff_mult": 2, "patch_budgets": dict(PATCH_BUDGETS)}
    cfg.update(overrides)
    return cfg


def build_vision_adapter(vcfg: Dict, text_dim: int, pretrained: bool) -> VisionAdapter:
    """With `pretrained`, load the tower from the Hub (training); otherwise build it empty for a checkpoint load."""
    if pretrained:
        from transformers import Siglip2VisionModel
        tower = Siglip2VisionModel.from_pretrained(vcfg["encoder"])
        vcfg = dict(vcfg)
        vcfg["tower_config"] = {k: v for k, v in tower.config.to_dict().items()
                                if k not in ("architectures", "transformers_version", "torch_dtype", "dtype")}
        return VisionAdapter(vcfg, text_dim, tower=tower)
    return VisionAdapter(vcfg, text_dim, no_init=True)


def encode_images(adapter: VisionAdapter, images: List[Any], budgets: List[int], device) -> torch.Tensor:
    batch = preprocess(images, budgets, adapter.patch_size)
    return adapter(**{k: v.to(device) for k, v in batch.items()})


def splice_positions(img_pos: torch.Tensor, slots: int) -> torch.Tensor:
    return img_pos[:, None] + torch.arange(slots, device=img_pos.device)[None, :]


__all__ = ["IMAGE_TOKENS", "IMAGE_SLOTS", "PATCH_BUDGETS", "VisionAdapter", "Resampler", "load_image",
           "is_image_state", "choose_budget", "preprocess", "patchify", "pad_patches", "encode_images", "build_vision_adapter",
           "vision_config_for", "splice_positions", "fit_feature_stats"]
