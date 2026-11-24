# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "torchvision", "transformers==4.47.1", "timm", "kornia", "einops",
#   "numpy", "opencv-python-headless", "pillow", "huggingface_hub", "requests",
# ]
# ///
"""Clean Food Filter: standard produce photos (Colin 2026-11-15 / 2026-11-24). Runs as a Hugging Face GPU job
(the workspace cannot reach Wikimedia); pass this file inline to hf_jobs "uv" with secrets HF_TOKEN.

Input: env PICKS_B64 = zlib+base64 JSON list of {item_id, slug, src, t, k, crop?, plain?}: candidate k of target t in the
candidate list `src` (produce/<src>/candidates.json in the dataset, made by the Commons search job; openly licensed files
picked by eye). crop = [x0, y0, x1, y1] fractions of the photo, to keep one fruit of several; meta_only = only look up the credit
fields (uploader, required attribution) for a photo made in an earlier run. Per photo: BiRefNet cutout
onto white (the photo's own pixels; nothing is drawn or generated), square with a margin, 800 x 800 WebP; plain = only
trim the even border and square-pad. Output: dataset DrNimbus/cff-photo-ai, produce/<TAG>/ (<item>/<slug>.webp, review
sheets, picks.json with each file's credit fields, all.zip) and a signed download link for all.zip ("ZIPURL ...").
"""
import io, json, os, time, zipfile, base64, zlib
import numpy as np, cv2, requests, torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from huggingface_hub import HfApi

from huggingface_hub import hf_hub_download
REPO = "DrNimbus/cff-photo-ai"; OUT = "/tmp/out"; TAG = os.environ.get("TAG", "final")
PICKS = json.loads(zlib.decompress(base64.b64decode(os.environ["PICKS_B64"])).decode())
CANDS = {src: json.load(open(hf_hub_download(REPO, f"produce/{src}/candidates.json", repo_type="dataset")))
         for src in {p["src"] for p in PICKS}}
SEL = []
for p in PICKS:
    c = CANDS[p["src"]][p["t"]]["cands"][p["k"]]
    SEL.append({**p, **{k: c[k] for k in ("title", "url", "w", "h", "page", "licence", "licence_url", "artist", "credit", "desc")}})
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, len(SEL), "photos", flush=True)
S = requests.Session(); S.headers["User-Agent"] = "CleanFoodFilter-produce-photos/0.1 (open-licence photo selection; https://clean-food-filter.vercel.app)"

# uploader and any required attribution text, for files whose Author field is empty or says "myself"
for i in range(0, len(SEL), 40):
    part = SEL[i:i + 40]
    try:
        r = S.get("https://commons.wikimedia.org/w/api.php", timeout=60, params=dict(
            action="query", format="json", prop="imageinfo", iiprop="user|extmetadata", iiextmetadatafilter="Attribution|AttributionRequired",
            titles="|".join(s["title"] for s in part)))
        pages = {p["title"]: p for p in r.json()["query"]["pages"].values()}
        norm = {n["from"]: n["to"] for n in r.json()["query"].get("normalized", [])}
        for s in part:
            ii = (pages.get(norm.get(s["title"], s["title"]), {}).get("imageinfo") or [{}])[0]; md = ii.get("extmetadata") or {}
            s["uploader"] = ii.get("user", "")
            s["attribution"] = (md.get("Attribution") or {}).get("value", "")
    except Exception as e:
        print("uploader lookup failed", e, flush=True)

from transformers import AutoModelForImageSegmentation
BR = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(DEV).eval()
if DEV == "cuda": BR = BR.half()
DT = torch.float16 if DEV == "cuda" else torch.float32
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def prob(rgb):
    x = torch.from_numpy(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA)).to(DEV).permute(2, 0, 1)[None].float() / 255
    x = ((x - MEAN) / STD).to(DT)
    o = BR(x)[-1].sigmoid().float()[0, 0].cpu().numpy()
    return cv2.resize(o, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def biggest(m, keep=0.15):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return m
    big = st[1:, 4].max(); out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, 4] >= keep * big: out[lab == i] = 1
    return out


def square(c):
    h, w = c.shape[:2]; side = int(max(h, w) * 1.12)
    cv = np.full((side, side, 3), 255, np.float32); oy, ox = (side - h) // 2, (side - w) // 2
    cv[oy:oy + h, ox:ox + w] = c
    return Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8)).resize((800, 800), Image.LANCZOS)


def cutout(rgb):
    p = prob(rgb)
    m = biggest((p > 0.5).astype(np.uint8))
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return None, "no object"
    near = cv2.dilate(m, np.ones((9, 9), np.uint8))
    a = np.clip((p - 0.2) / 0.6, 0, 1) * near.astype(np.float32)
    a = np.maximum(a, cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32))
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    comp = rgb.astype(np.float32) * a[..., None] + 255 * (1 - a[..., None])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    touch = float(np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean())
    return square(comp[y0:y1, x0:x1]), f"area {m.mean():.2f} touch {touch:.2f}"


def plain(rgb):
    """already on a plain light background: trim the even border, square-pad onto white"""
    g = rgb.astype(np.int16); bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]), 0)
    ys, xs = np.where(np.abs(g - bg).max(2) > 24)
    c = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if len(ys) else rgb
    return square(c.astype(np.float32)), "plain"


def thumb_url(u, w=1600):
    """upload.wikimedia.org/wikipedia/commons/a/ab/F.jpg -> .../commons/thumb/a/ab/F.jpg/1600px-F.jpg"""
    k = u.split("/wikipedia/commons/")
    if len(k) != 2: return u
    return f"{k[0]}/wikipedia/commons/thumb/{k[1]}/{w}px-{k[1].split('/')[-1]}"


def fetch(s):
    for u in ([thumb_url(s["url"])] if s.get("w", 9999) > 1700 else []) + [s["url"]]:
        for k in range(4):
            try:
                r = S.get(u, timeout=60)
                if r.status_code == 200:
                    im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
                    if max(im.size) > 1800: im.thumbnail((1800, 1800), Image.LANCZOS)
                    return np.asarray(im)
                if r.status_code in (429, 503): time.sleep(3 + 4 * k); continue
                break
            except Exception as e:
                print("fetch error", u, e, flush=True); time.sleep(3)
    return None


os.makedirs(f"{OUT}/sheets", exist_ok=True)
log, done = [], []
for n, s in enumerate(SEL):
    key = f"{s['item_id']}/{s['slug']}"
    if s.get("meta_only"): continue  # credit fields only (photo made in an earlier run)
    rgb = fetch(s)
    if rgb is None: log.append((key, "download failed")); print(n, key, "download failed", flush=True); continue
    if s.get("crop"):
        h, w = rgb.shape[:2]; x0, y0, x1, y1 = s["crop"]
        rgb = np.ascontiguousarray(rgb[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)])
    img, note = plain(rgb) if s.get("plain") else cutout(rgb)
    if img is None: log.append((key, note)); print(n, key, note, flush=True); continue
    os.makedirs(f"{OUT}/{s['item_id']}", exist_ok=True)
    img.save(f"{OUT}/{key}.webp", "WEBP", quality=86, method=6)
    done.append((key, img)); log.append((key, note)); print(n, key, note, flush=True)
    time.sleep(0.4)

try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
except Exception: F = ImageFont.load_default()
T = 300
for s in range(0, len(done), 20):  # review sheets: 5 x 4 tiles, grey between tiles so the white edge shows
    part = done[s:s + 20]; rows = (len(part) + 4) // 5
    sh = Image.new("RGB", (5 * T, rows * (T + 22)), (190, 190, 190)); d = ImageDraw.Draw(sh)
    for k, (key, img) in enumerate(part):
        x, y = (k % 5) * T, (k // 5) * (T + 22)
        sh.paste(img.resize((T - 6, T - 6)), (x + 3, y + 3)); d.text((x + 4, y + T), f"{s + k} {key}", fill="black", font=F)
    sh.save(f"{OUT}/sheets/r{s // 20:02d}.jpg", quality=85)
open(f"{OUT}/log.tsv", "w").write("\n".join(f"{a}\t{b}" for a, b in log) + "\n")
json.dump(SEL, open(f"{OUT}/picks.json", "w"), indent=1, ensure_ascii=False)
with zipfile.ZipFile("/tmp/all.zip", "w") as z:
    for key, _ in done: z.write(f"{OUT}/{key}.webp", f"{key}.webp")
    z.write(f"{OUT}/log.tsv", "log.tsv"); z.write(f"{OUT}/picks.json", "picks.json")
os.rename("/tmp/all.zip", f"{OUT}/all.zip")
HfApi().upload_folder(folder_path=OUT, path_in_repo=f"produce/{TAG}", repo_id=REPO, repo_type="dataset", commit_message=f"produce photos {TAG}")
r = requests.head(f"https://huggingface.co/datasets/{REPO}/resolve/main/produce/{TAG}/all.zip",
                  headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}, allow_redirects=False)
print("ZIPURL", r.headers.get("Location"), flush=True)
print("done", len(done), "of", len(SEL), flush=True)
