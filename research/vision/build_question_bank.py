"""Expand question_bank.TEMPLATES with paraphrases and translations (one teacher call per template).

Writes question_bank.json: {"<domain>.<id>": {"paraphrases": [2 x str],
                                             "translations": {lang: {"instructions": str, "criteria": ...}}}}
Choice keys never change (they are the API labels); only their descriptions are translated.

    python research/vision/build_question_bank.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import question_bank as qbank  # noqa: E402
from openrouter import Client, message_json  # noqa: E402

LANGS = {"de": "German", "es": "Spanish", "fr": "French", "hi": "Hindi", "ja": "Japanese",
         "zh": "Simplified Chinese", "ar": "Arabic", "pt": "Portuguese"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "question_bank.json")

SCHEMA = {
    "type": "object",
    "properties": {
        "paraphrases": {"type": "array", "items": {"type": "string"}},
        "translations": {"type": "array", "items": {
            "type": "object",
            "properties": {"lang": {"type": "string"}, "instructions": {"type": "string"},
                           "criteria": {"type": "array", "items": {"type": "string"}}},
            "required": ["lang", "instructions", "criteria"], "additionalProperties": False}},
    },
    "required": ["paraphrases", "translations"],
    "additionalProperties": False,
}


def criteria_texts(t):
    if t["type"] == "choice":
        # an empty description is the key itself, spelled out, so the translation has something to carry
        return [v if v else k.replace("_", " ") for k, v in t["criteria"].items()]
    if t["type"] == "score":
        return list(t["criteria"])
    return []


async def one(client, domain, idx, t, bank):
    key = qbank.template_key(domain, t)
    if key in bank:
        return
    langs = sorted(LANGS)
    pick = [langs[(idx * 2) % len(langs)], langs[(idx * 2 + 1) % len(langs)]]
    crit = criteria_texts(t)
    prompt = (
        "Here is a question that will be asked about an image, with its answer options.\n"
        "Question: %s\nOptions: %s\n\n"
        "1. Write 2 English paraphrases of the question that keep exactly the same meaning and answer. "
        "Vary wording and structure. Keep any {obj} placeholder verbatim.\n"
        "2. Translate the question and each option text into: %s. Keep {obj} verbatim. Return the option "
        "texts in the same order, one per option (%d options).\n"
        "Return JSON: {\"paraphrases\": [...], \"translations\": [{\"lang\": code, \"instructions\": ..., "
        "\"criteria\": [...]}]} using language codes %s."
    ) % (t["instructions"], json.dumps(crit, ensure_ascii=False) if crit else "(yes/no)",
         ", ".join(LANGS[x] for x in pick), len(crit), pick)
    for _ in range(3):
        data = await client.chat([{"role": "user", "content": prompt}], schema=SCHEMA, reasoning_effort="low")
        parsed = message_json(data) if data else None
        if not parsed:
            continue
        paras = [p.strip() for p in parsed.get("paraphrases", []) if isinstance(p, str) and p.strip()][:2]
        if "{obj}" in t["instructions"] and any("{obj}" not in p for p in paras):
            continue
        trans = {}
        ok = True
        for tr in parsed.get("translations", []):
            lang = tr.get("lang")
            if lang not in pick:
                continue
            if "{obj}" in t["instructions"] and "{obj}" not in tr["instructions"]:
                ok = False
            if t["type"] != "noul" and len(tr["criteria"]) != len(crit):
                ok = False
            entry = {"instructions": tr["instructions"].strip()}
            if t["type"] == "choice":
                entry["criteria"] = dict(zip(t["criteria"].keys(), [c.strip() for c in tr["criteria"]]))
            elif t["type"] == "score":
                entry["criteria"] = [c.strip() for c in tr["criteria"]]
            trans[lang] = entry
        if ok and len(paras) == 2 and set(trans) == set(pick):
            bank[key] = {"paraphrases": paras, "translations": trans}
            return
    print("failed", key)


async def main():
    bank = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            bank = json.load(f)
    client = Client(concurrency=64)
    jobs = [one(client, d, i, t, bank) for d, ts in qbank.TEMPLATES.items() for i, t in enumerate(ts)]
    await asyncio.gather(*jobs)
    await client.close()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(bank, f, ensure_ascii=False, indent=1)
    print("templates expanded: %d / %d" % (len(bank), sum(len(v) for v in qbank.TEMPLATES.values())))


if __name__ == "__main__":
    asyncio.run(main())
