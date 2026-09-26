"""Build the blind audit page: one self-contained HTML file with the audit items embedded.

For each item you see the image and the question and pick the option that is correct (or
"unsure / bad question"). The teacher's answer is never shown, so it cannot anchor you. Progress
is kept in the browser; "Export verdicts" downloads audit_verdicts.json for eval_vision.py --audit.

    python research/vision/build_audit_page.py --items E:/LAYA-V/data/items/audit_items.jsonl \
        --image-root E:/LAYA-V/data --out E:/LAYA-V/data/audit.html
"""
import argparse
import base64
import html
import io
import json
import os

from PIL import Image


def options(q):
    if q["type"] == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in q["criteria"].items()]
    if q["type"] == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(q["criteria"])]
    return ["no", "yes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.items, encoding="utf-8")]
    data = []
    for i, it in enumerate(items):
        path = os.path.join(args.image_root, it["image"])
        if not os.path.exists(path):
            path = os.path.join(args.image_root, "pool", it["image"])
        img = Image.open(path).convert("RGB")
        img.thumbnail((1100, 1100))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data.append({"i": i, "domain": it["domain"], "q": it["q"]["instructions"], "type": it["q"]["type"],
                     "opts": options(it["q"]), "img": base64.b64encode(buf.getvalue()).decode()})
    page = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__N__", str(len(data)))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", args.out, len(data), "items")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laya-V label audit</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deddd8;--acc:#2f5bd3;--ok:#1e7b4a}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#eceae4;--muted:#9a988f;--card:#20201e;--line:#34332f;--acc:#7c9cf5;--ok:#58c08a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:1}
header b{font-size:16px}.bar{flex:1;min-width:120px;height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar i{display:block;height:100%;background:var(--ok)}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:20px}
@media (max-width:820px){main{grid-template-columns:1fr}}
img{width:100%;border-radius:8px;border:1px solid var(--line);background:var(--card)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.meta{color:var(--muted);font-size:13px;margin-bottom:6px}.q{font-size:18px;margin:0 0 14px}
button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:9px 12px;cursor:pointer;text-align:left;width:100%;margin:0 0 8px}
button:hover{border-color:var(--acc)}button.sel{border-color:var(--ok);outline:2px solid var(--ok)}
kbd{font-size:12px;color:var(--muted);margin-right:8px}.nav{display:flex;gap:8px}.nav button{width:auto;text-align:center}
</style></head><body>
<header><b>Laya-V label audit</b><span id="count"></span><div class="bar"><i id="prog"></i></div>
<div class="nav"><button onclick="go(-1)">&larr; Prev</button><button onclick="go(1)">Next &rarr;</button>
<button onclick="nextOpen()">Next unanswered</button><button onclick="exportV()">Export verdicts</button></div></header>
<main><div><img id="img" alt="audit image"></div>
<div class="card"><div class="meta" id="meta"></div><p class="q" id="q"></p><div id="opts"></div>
<p class="meta">Pick the option that is actually correct for this image. Keys: 1&ndash;9 pick, 0 unsure, &larr;/&rarr; move.</p></div></main>
<script>
const D=__DATA__;const KEY="laya-v-audit";let V={};try{V=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let cur=0;
function save(){try{localStorage.setItem(KEY,JSON.stringify(V))}catch(e){}}
function render(){const d=D[cur];document.getElementById("img").src="data:image/jpeg;base64,"+d.img;
document.getElementById("meta").textContent=`#${cur+1} of __N__ · ${d.domain} · ${d.type}`;document.getElementById("q").textContent=d.q;
const box=document.getElementById("opts");box.innerHTML="";
d.opts.forEach((o,j)=>{const b=document.createElement("button");b.innerHTML=`<kbd>${j+1}</kbd>`;b.appendChild(document.createTextNode(o));
if(V[d.i]===j)b.className="sel";b.onclick=()=>pick(j);box.appendChild(b)});
const u=document.createElement("button");u.innerHTML="<kbd>0</kbd>unsure / bad question";if(V[d.i]===null)u.className="sel";u.onclick=()=>pick(null);box.appendChild(u);
const n=Object.keys(V).length;document.getElementById("count").textContent=`${n} / __N__ done`;document.getElementById("prog").style.width=(100*n/D.length)+"%"}
function pick(j){V[D[cur].i]=j;save();if(cur<D.length-1)cur++;render()}
function go(s){cur=Math.max(0,Math.min(D.length-1,cur+s));render()}
function nextOpen(){for(let k=1;k<=D.length;k++){const c=(cur+k)%D.length;if(!(D[c].i in V)){cur=c;render();return}}}
function exportV(){const out=D.map(d=>({i:d.i,human:(d.i in V)?V[d.i]:undefined})).filter(x=>x.human!==undefined);
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(out)],{type:"application/json"}));a.download="audit_verdicts.json";a.click()}
document.addEventListener("keydown",e=>{if(e.key>="1"&&e.key<="9"){const j=+e.key-1;if(j<D[cur].opts.length)pick(j)}else if(e.key==="0")pick(null);
else if(e.key==="ArrowRight")go(1);else if(e.key==="ArrowLeft")go(-1)});
nextOpen();render();
</script></body></html>"""

if __name__ == "__main__":
    main()
