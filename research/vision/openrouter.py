"""Minimal async OpenRouter chat client shared by the Laya-V labelling scripts."""
import asyncio
import base64
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
TEACHER = os.environ.get("LAYA_V_TEACHER", "stealth/space-bunny-alpha")


def image_part(path: str, max_side: int = 2048, max_bytes: int = 4_000_000) -> Dict[str, Any]:
    """Image as a data URL with its real media type. Images over `max_side` px or `max_bytes` are re-encoded
    as JPEG at <= max_side (some providers reject larger ones; Laya-V itself sees at most 1,024 patches)."""
    import io

    from PIL import Image

    with open(path, "rb") as f:
        raw = f.read()
    mime = ("image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/gif" if raw[:4] == b"GIF8"
            else "image/webp" if raw[8:12] == b"WEBP" else "image/jpeg")
    with Image.open(io.BytesIO(raw)) as im:
        if len(raw) > max_bytes or max(im.size) > max_side:
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=90)
            raw, mime = buf.getvalue(), "image/jpeg"
    return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())}}


class Client:
    def __init__(self, concurrency: int = 64, model: str = TEACHER, timeout: float = 180.0):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("set OPENROUTER_API_KEY")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.http = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": "Bearer " + key, "HTTP-Referer": "https://github.com/NandhaKishorM/laya",
                     "X-Title": "laya-v labelling"},
            limits=httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency),
        )
        self.stats = {"ok": 0, "http_err": 0, "429": 0, "latency": [], "usd": 0.0, "schema_fallback": 0}

    async def close(self):
        await self.http.aclose()

    async def chat(self, messages: List[Dict], schema: Optional[Dict] = None, reasoning_effort: str = "low",
                   max_tokens: int = 4096, temperature: float = 0.0, retries: int = 5) -> Optional[Dict]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                                "temperature": temperature, "usage": {"include": True}}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort, "exclude": True}
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answers", "strict": True, "schema": schema}}
        for attempt in range(retries):
            async with self.sem:
                t0 = time.perf_counter()
                try:
                    r = await self.http.post(URL, json=body)
                except httpx.HTTPError as e:
                    self.stats["http_err"] += 1
                    err = repr(e)
                    r = None
            if r is not None:
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except json.JSONDecodeError:
                        data = None
                    if data and data.get("choices"):
                        self.stats["ok"] += 1
                        self.stats["latency"].append(time.perf_counter() - t0)
                        self.stats["usd"] += float((data.get("usage") or {}).get("cost") or 0.0)
                        return data
                    err = "bad body: %s" % r.text[:300]
                else:
                    if r.status_code == 429:
                        self.stats["429"] += 1
                    else:
                        self.stats["http_err"] += 1
                    err = "HTTP %d: %s" % (r.status_code, r.text[:300])
                    if r.status_code == 400 and "response_format" in body:
                        # Some providers reject json_schema; retry once with the prompt alone
                        # (message_json tolerates free-form JSON).
                        self.stats["schema_fallback"] += 1
                        body.pop("response_format")
                        continue
                    if r.status_code in (400, 401, 402, 403, 404):
                        print("fatal-ish:", err)
                        if r.status_code in (401, 402, 403, 404):
                            raise SystemExit(err)
                        return None
            await asyncio.sleep(min(60, 2 ** attempt + random.random() * 2))
        print("giving up:", err)
        return None


def message_json(data: Dict) -> Optional[Any]:
    """Parse the JSON body of a chat completion, tolerating code fences."""
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None
