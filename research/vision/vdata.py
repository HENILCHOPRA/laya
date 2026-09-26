"""Laya-V training data: item format, dataset, batch sampler and collate.

Item files are JSONL, one row per question:
    {"gid": str,                 # rows with the same gid share one image (and one tower pass)
     "image": str | null,        # path relative to the image root; null for text-only rows
     "domain": str,              # photo | document | receipt | chart | screenshot | meme | text
     "state": str | dict,        # text state (caption, note, JSON fields); "" if none
     "q": {"type", "instructions", "criteria"?, "labels"?},
     "target": [float, ...],     # soft target in Laya label order
     "source": str,              # provenance: teacher:space-bunny | public:<dataset> | distill:laya-multilingual
     "split": "train" | "calib" | "eval",
     "task": str}                # free-form slice name for evaluation
"""
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from laya.agent import Agent  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402
from laya.vision import IMAGE_SLOTS, load_image, pad_patches, patchify  # noqa: E402

DOMAIN_BUDGET = {"photo": 256, "meme": 256, "chart": 576, "screenshot": 576, "document": 1024, "receipt": 1024}


def read_items(paths: List[str], split: Optional[str] = None) -> List[Dict]:
    items = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                it = json.loads(line)
                if split is None or it.get("split", "train") == split:
                    items.append(it)
    return items


def holdout_keys(teacher_items_path: str) -> set:
    """Source-image keys of teacher eval/calib images: no training row from any source may use them."""
    keys = set()
    with open(teacher_items_path, encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            if it.get("split") in ("eval", "calib") and it.get("srckey"):
                keys.add(it["srckey"])
    return keys


def group_items(items: List[Dict]) -> List[List[Dict]]:
    groups = defaultdict(list)
    for it in items:
        groups[it["gid"]].append(it)
    return list(groups.values())


def budget_of(row: Dict) -> int:
    if not row.get("image"):
        return 0
    return int(row.get("budget") or DOMAIN_BUDGET.get(row.get("domain"), 576))


class GroupDataset(Dataset):
    """One element = one image (or text state) with all its question rows, tokenized."""

    def __init__(self, groups: List[List[Dict]], tok, image_root: str, max_len: int = 1024,
                 head_max_len: int = 256, patch_size: int = 16, max_rows_per_group: int = 12,
                 image_override=None):
        self.groups = groups
        self.tok = tok
        self.image_root = image_root
        self.max_len, self.head_max_len = max_len, head_max_len
        self.patch_size = patch_size
        self.max_rows = max_rows_per_group
        # For failure probes: a callable (gid, PIL image) -> PIL image applied before patchify.
        self.image_override = image_override

    def __len__(self):
        return len(self.groups)

    def row_count(self, i):
        return min(len(self.groups[i]), self.max_rows)

    def budget(self, i):
        return budget_of(self.groups[i][0])

    def __getitem__(self, i):
        rows = self.groups[i][: self.max_rows]
        first = rows[0]
        out = {"gid": first["gid"], "patches": None, "shape": None, "rows": []}
        has_image = bool(first.get("image"))
        if has_image:
            img = load_image(os.path.join(self.image_root, first["image"]))
            if self.image_override is not None:
                img = self.image_override(first["gid"], img)
            out["patches"], out["shape"] = patchify(img, budget_of(first), self.patch_size)
        state = first.get("state", "")
        truncate_left = isinstance(state, list)
        for r in rows:
            q = Agent._to_internal(r["q"])
            if has_image:
                ids, markers, img_pos = build_sequence(self.tok, r.get("state", ""), q, self.max_len,
                                                       self.head_max_len, truncate_left=truncate_left,
                                                       image_slots=IMAGE_SLOTS)
            else:
                ids, markers = build_sequence(self.tok, r.get("state", ""), q, self.max_len, self.head_max_len,
                                              truncate_left=truncate_left)
                img_pos = -1
            k = len(render_options(q))
            if len(markers) != k or len(r["target"]) != k:
                continue
            out["rows"].append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]], "target": r["target"],
                                "img_pos": img_pos, "meta": {"source": r.get("source"), "task": r.get("task"),
                                                             "domain": r.get("domain"), "gid": r["gid"],
                                                             "qkey": r.get("qkey")}})
        return out


class BudgetBatchSampler(Sampler):
    """Packs groups into batches of at most `max_rows` question rows, never mixing patch budgets,
    so the tower pads little. Reshuffles every epoch (set_epoch)."""

    def __init__(self, ds: GroupDataset, max_rows: int, shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.ds, self.max_rows, self.shuffle, self.seed, self.drop_last = ds, max_rows, shuffle, seed, drop_last
        self.epoch = 0
        self._batches = self._make()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._batches = self._make()

    def _make(self):
        rng = random.Random(self.seed + self.epoch)
        by_budget = defaultdict(list)
        for i in range(len(self.ds)):
            by_budget[self.ds.budget(i)].append(i)
        batches = []
        for _, idxs in sorted(by_budget.items()):
            if self.shuffle:
                rng.shuffle(idxs)
            cur, rows = [], 0
            for i in idxs:
                n = self.ds.row_count(i)
                if cur and rows + n > self.max_rows:
                    batches.append(cur)
                    cur, rows = [], 0
                cur.append(i)
                rows += n
            if cur and not self.drop_last:
                batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def collate_groups(groups: List[Dict], pad_id: int) -> Dict:
    rows, seqs, shapes, image_of_group = [], [], [], []
    for g in groups:
        idx = -1
        if g["patches"] is not None:
            idx = len(seqs)
            seqs.append(g["patches"])
            shapes.append(g["shape"])
        for r in g["rows"]:
            rows.append((r, idx))
    n = len(rows)
    L = max(len(r["ids"]) for r, _ in rows)
    kmax = max(len(r["markers"]) for r, _ in rows)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, (r, _) in enumerate(rows):
        ids[i, : len(r["ids"])] = torch.tensor(r["ids"])
        att[i, : len(r["ids"])] = 1
        k = len(r["markers"])
        mpos[i, :k] = torch.tensor(r["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(r["target"], dtype=torch.float32)
    b = {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
         "qtype": torch.tensor([r["qtype"] for r, _ in rows]),
         "img_pos": torch.tensor([r["img_pos"] for r, _ in rows], dtype=torch.long),
         "vis_index": torch.tensor([gi for _, gi in rows], dtype=torch.long),
         "meta": [r["meta"] for r, _ in rows]}
    b["pixels"] = pad_patches(seqs, shapes) if seqs else None
    return b


def model_forward(model, b, device, vis_tokens=None):
    """Forward a collated batch; runs the vision adapter when the batch has images."""
    extra = {}
    if b["pixels"] is not None:
        if vis_tokens is None:
            vis_tokens = model.vision(**{k: v.to(device, non_blocking=True) for k, v in b["pixels"].items()})
        extra = {"vis_tokens": vis_tokens, "vis_index": b["vis_index"].to(device), "img_pos": b["img_pos"].to(device)}
    return model(b["input_ids"].to(device, non_blocking=True), b["attention_mask"].to(device, non_blocking=True),
                 b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device), **extra)


def softmax_np(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
