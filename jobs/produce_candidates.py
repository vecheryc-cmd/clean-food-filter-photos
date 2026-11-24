# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "pillow", "numpy", "huggingface_hub"]
# ///
"""Clean Food Filter: find openly licensed Wikimedia Commons photos for the standard produce list.
For each target (item + variety) runs the Commons searches, keeps files whose licence is public domain, CC0,
CC BY or CC BY-SA, ranks them (plain light background first), and writes numbered contact sheets for review.
Runs as a Hugging Face CPU job (the workspace cannot reach Wikimedia): env TARGETS_B64 = zlib+base64 JSON list of
{item_id, slug, subject, queries}; env OUTDIR (default "candidates"). Output: dataset DrNimbus/cff-photo-ai,
produce/<OUTDIR>/ (candidates.json, sheets/sNN.jpg: 4 targets per sheet, numbered tiles with licence and width)."""
import io, json, os, re, sys, time, concurrent.futures as cf, base64, zlib
import numpy as np, requests
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import HfApi

TARGETS = json.loads(zlib.decompress(base64.b64decode(os.environ["TARGETS_B64"])).decode())
REPO = "DrNimbus/cff-photo-ai"; OUT = "/tmp/out"; os.makedirs(f"{OUT}/sheets", exist_ok=True)
S = requests.Session(); S.headers["User-Agent"] = "CleanFoodFilter-produce-photos/0.1 (open-licence photo selection; https://clean-food-filter.vercel.app)"
API = "https://commons.wikimedia.org/w/api.php"
OK = re.compile(r"^(public domain|pd\b|pd-|cc0|cc[- ]zero|cc by(-sa)? ?\d|cc-by(-sa)?-\d|cc by(-sa)?$|cc-by(-sa)?$)", re.I)
BAD_LIC = re.compile(r"\bnc\b|\bnd\b|non-?commercial|no ?deriv", re.I)
BADW = re.compile(r"market|stall|shop|store|tree|plant|flower|blossom|field|farm|garden|dish|salad|recipe|soup|cake|pie\b|juice|slice|sliced|half|halves|cut\b|cross.?section|interior|inside|seeds?\b|leaf spot|disease|pest|map|logo|diagram|drawing|illustration|painting|stamp|chart|harvest|people|woman|man\b|child|hand", re.I)
GOODW = re.compile(r"white background|isolated|on white|cutout|studio", re.I)

def get(url, **kw):
    for k in range(5):
        try:
            r = S.get(url, timeout=40, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 + 3 * k); continue
            return r
        except Exception:
            time.sleep(2 + 3 * k)
    return None

def search(q):
    p = dict(action="query", format="json", generator="search", gsrsearch=f"{q} filetype:bitmap", gsrnamespace=6, gsrlimit=40,
             prop="imageinfo", iiprop="url|size|mime|extmetadata", iiurlwidth=330,
             iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist|Credit|ImageDescription|ObjectName")
    r = get(API, params=p)
    if not r or r.status_code != 200: return []
    pages = (r.json().get("query") or {}).get("pages") or {}
    out = []
    for pg in sorted(pages.values(), key=lambda x: x.get("index", 99)):
        ii = (pg.get("imageinfo") or [{}])[0]; md = ii.get("extmetadata") or {}
        lic = (md.get("LicenseShortName") or {}).get("value", "")
        if not OK.search(lic.strip()) or BAD_LIC.search(lic): continue
        if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"): continue
        if min(ii.get("width", 0), ii.get("height", 0)) < 600: continue
        out.append(dict(title=pg["title"], rank=pg.get("index", 99), w=ii["width"], h=ii["height"], mime=ii["mime"],
                        url=ii["url"], thumb=ii.get("thumburl"), page=ii.get("descriptionurl"), licence=lic,
                        licence_url=(md.get("LicenseUrl") or {}).get("value", ""),
                        artist=re.sub(r"<[^>]+>", "", (md.get("Artist") or {}).get("value", "")).strip(),
                        credit=re.sub(r"<[^>]+>", "", (md.get("Credit") or {}).get("value", "")).strip()[:200],
                        desc=re.sub(r"<[^>]+>", "", (md.get("ImageDescription") or {}).get("value", "")).strip()[:300]))
    return out

def bgscore(img):
    a = np.asarray(img.convert("RGB").resize((160, 160)), np.float32); b = 10
    band = np.concatenate([a[:b].reshape(-1, 3), a[-b:].reshape(-1, 3), a[:, :b].reshape(-1, 3), a[:, -b:].reshape(-1, 3)])
    L = band.mean(1); return float(L.mean()), float(band.std(0).mean())

def thumb(c):
    r = get(c["thumb"]) if c.get("thumb") else None
    if not r or r.status_code != 200: return None
    try: return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception: return None

results = []
for n, t in enumerate(TARGETS):
    seen, cands = set(), []
    for q in t["queries"]:
        for c in search(q):
            if c["title"] in seen: continue
            seen.add(c["title"]); cands.append(c)
        time.sleep(0.3)
    # pre-rank by words, then fetch thumbs for the best 26
    for c in cands:
        s = 0.0; tl = c["title"] + " " + c["desc"][:120]
        if GOODW.search(tl): s += 3
        if BADW.search(c["title"]): s -= 2
        s -= 0.05 * c["rank"]; c["pre"] = s
    cands.sort(key=lambda c: -c["pre"]); cands = cands[:26]
    with cf.ThreadPoolExecutor(4) as ex: ths = list(ex.map(thumb, cands))
    keep = []
    for c, im in zip(cands, ths):
        if im is None: continue
        m, sd = bgscore(im); c["bg_mean"], c["bg_std"] = round(m, 1), round(sd, 1)
        c["score"] = c["pre"] + (4 if m > 225 and sd < 15 else 2 if sd < 15 else 0) + (1 if m > 200 else 0)
        c["_im"] = im; keep.append(c)
    keep.sort(key=lambda c: -c["score"]); keep = keep[:12]
    t["cands"] = keep; results.append(t)
    print(n, t["item_id"], t["slug"], len(cands), "->", len(keep), flush=True)

# contact sheets: 4 targets per sheet, 9 tiles per row
TW, TH, LH = 150, 150, 34
try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except Exception: F = ImageFont.load_default()
for s in range(0, len(results), 4):
    rows = results[s:s + 4]
    sh = Image.new("RGB", (12 * TW, len(rows) * (TH + LH)), "white"); d = ImageDraw.Draw(sh)
    for r, t in enumerate(rows):
        y = r * (TH + LH)
        d.rectangle([0, y, 12 * TW, y + 18], fill=(30, 30, 30)); d.text((4, y + 1), f"T{results.index(t)} {t['item_id']}/{t['slug']}: {t['subject']}", fill="white", font=F)
        for k, c in enumerate(t["cands"]):
            im = c["_im"].copy(); im.thumbnail((TW - 6, TH - 6))
            x0 = k * TW + (TW - im.width) // 2; y0 = y + 18 + (TH - im.height) // 2
            sh.paste(im, (x0, y0))
            lic = c["licence"].replace("Public domain", "PD")[:12]
            d.rectangle([k * TW, y + 18, k * TW + 22, y + 36], fill=(255, 230, 0)); d.text((k * TW + 4, y + 19), str(k), fill="black", font=F)
            d.text((k * TW + 2, y + 18 + TH - 2), f"{lic} {c['w']}", fill=(90, 90, 90), font=F)
    sh.save(f"{OUT}/sheets/s{s // 4:02d}.jpg", quality=82)
for t in results:
    for c in t["cands"]: c.pop("_im", None)
json.dump(results, open(f"{OUT}/candidates.json", "w"), indent=1, ensure_ascii=False)
api = HfApi(); api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=OUT, path_in_repo=f"produce/{os.environ.get('OUTDIR', 'candidates')}", repo_id=REPO, repo_type="dataset", commit_message="produce photo candidates")
print("done", len(results), "targets", sum(len(t["cands"]) for t in results), "candidates", flush=True)
