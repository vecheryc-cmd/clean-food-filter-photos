# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "torchvision", "transformers==4.47.1", "timm", "kornia", "einops",
#   "numpy", "opencv-python-headless", "pillow", "huggingface_hub", "spandrel", "requests",
# ]
# ///
"""Clean Food Filter: product-photo clean-up on a Hugging Face GPU job.

Per photo: 1) BiRefNet cutout onto white (pack pixels are the photo's own pixels; nothing is redrawn),
2) mild AI sharpening (Real-ESRGAN x2, blended with a plain resize so small print is not re-invented),
3) safe polish (levels, white balance, light unsharp, edge clean-up).
Writes both versions ("cut" and "enh") so each can be judged.  Output goes to a Hugging Face dataset
repo in shards of 500, with log.csv and numbered review sheets.  Re-running skips finished shards.
"""
import argparse, concurrent.futures as cf, csv, io, json, os, sys, time, zipfile
import numpy as np, cv2, requests, torch
from PIL import Image, ImageOps, ImageDraw
from huggingface_hub import HfApi

P = argparse.ArgumentParser()
P.add_argument("--waves", default="t120")          # comma list: t120, 01..12
P.add_argument("--src", default="https://raw.githubusercontent.com/vecheryc-cmd/clean-food-filter-photos/main/jobs/waves")
P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
P.add_argument("--shard", type=int, default=500)
P.add_argument("--limit", type=int, default=0)     # only the first N photos of each wave (pilot)
P.add_argument("--blend", type=float, default=0.55)
P.add_argument("--only-shards", default="")         # e.g. "0,1"
A = P.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
api = HfApi()
api.create_repo(A.repo, repo_type="dataset", private=True, exist_ok=True)
have = set(api.list_repo_files(A.repo, repo_type="dataset"))

# ---------- models ----------
from transformers import AutoModelForImageSegmentation
BR = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).to(DEV).eval()
if DEV == "cuda": BR = BR.half()
from spandrel import ModelLoader
ESR_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
if not os.path.exists("x2.pth"):
    open("x2.pth", "wb").write(requests.get(ESR_URL, timeout=120).content)
ESR = ModelLoader().load_from_file("x2.pth").model.to(DEV).eval()
if DEV == "cuda": ESR = ESR.half()
DT = torch.float16 if DEV == "cuda" else torch.float32
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)

@torch.no_grad()
def prob(rgb):
    x = torch.from_numpy(cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_AREA)).to(DEV).permute(2, 0, 1)[None].float() / 255
    x = ((x - MEAN) / STD).to(DT)
    o = BR(x)[-1].sigmoid().float()[0, 0].cpu().numpy()
    return cv2.resize(o, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

@torch.no_grad()
def esr(rgb8):
    x = torch.from_numpy(rgb8).to(DEV).permute(2, 0, 1)[None].to(DT) / 255
    y = ESR(x).clamp(0, 1).float()[0].permute(1, 2, 0).cpu().numpy()
    return y * 255

# ---------- cutout (same rules as the laptop job clean10.py) ----------
def skin(rgb):
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ((y[..., 1] > 138) & (y[..., 1] < 175) & (y[..., 2] > 85) & (y[..., 2] < 128)).astype(np.uint8)
def biggest(m, keep=0.25):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1: return m
    big = st[1:, 4].max(); out = np.zeros_like(m)
    for i in range(1, n):
        if st[i, 4] >= keep * big: out[lab == i] = 1
    return out
def fill(m):
    h, w = m.shape; ff = np.pad(m, 1).astype(np.uint8).copy(); mk = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, mk, (0, 0), 2); return (ff[1:-1, 1:-1] != 2).astype(np.uint8)
def cutout(rgb):
    p = prob(rgb); notes = []
    m = fill(biggest((p > 0.5).astype(np.uint8)))
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return None, None, "no object", {}
    sf = float(cv2.morphologyEx(skin(rgb) * m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).sum() / max(m.sum(), 1))
    if sf > 0.04: notes.append(f"hand-risk {sf:.0%}")
    touch = float(np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]).mean())
    near = cv2.dilate(m, np.ones((9, 9), np.uint8))
    a = np.clip((p - 0.2) / 0.6, 0, 1) * near.astype(np.float32)
    a = np.maximum(a, cv2.erode(m, np.ones((3, 3), np.uint8)).astype(np.float32))
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    comp = rgb.astype(np.float32) * a[..., None] + 255 * (1 - a[..., None])
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    c = comp[y0:y1, x0:x1]; ca = a[y0:y1, x0:x1]; h, w = c.shape[:2]; side = int(max(h, w) * 1.12)
    cv = np.full((side, side, 3), 255, np.float32); av = np.zeros((side, side), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2; cv[oy:oy + h, ox:ox + w] = c; av[oy:oy + h, ox:ox + w] = ca
    out = np.asarray(Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8)).resize((800, 800), Image.LANCZOS))
    al = cv2.resize(av, (800, 800), interpolation=cv2.INTER_AREA)
    return out, al, "; ".join(notes), dict(area=round(float(m.mean()), 3), touch=round(touch, 2), skin=round(sf, 2))

# ---------- enhancement ----------
def bg_mask(a):
    cand = (a.min(2) >= 253).astype(np.uint8)
    n, lab = cv2.connectedComponents(cand, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(border))
def lum(a): return a @ np.array([0.299, 0.587, 0.114], np.float32)
def polish(a8):   # "C1" polish from the 24-photo pilot (no shadow; shadows are added by the app's CSS)
    a = a8.astype(np.float32); bg = bg_mask(a8); prod = ~bg
    if prod.sum() < 500: return a8
    k = np.ones((3, 3), np.uint8); pu = prod.astype(np.uint8)
    inner2 = cv2.erode(pu, k, iterations=3).astype(bool); band = prod & ~cv2.erode(pu, k, iterations=2).astype(bool)
    if inner2.sum() < 100: return a8
    dist, idx = cv2.distanceTransformWithLabels((~inner2).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.where(inner2); ly = np.zeros(idx.max() + 1, int); lx = np.zeros(idx.max() + 1, int)
    ly[idx[ys, xs]] = ys; lx[idx[ys, xs]] = xs
    nb = cv2.GaussianBlur(a, (0, 0), 2.0); near = 0.5 * a[ly[idx], lx[idx]] + 0.5 * nb[ly[idx], lx[idx]]
    d_in = 255 - near; d_px = 255 - a
    al = np.clip(np.sum(d_px * d_in, 2) / (np.sum(d_in * d_in, 2) + 1e-3), 0, 1)
    clean = al[..., None] * near + (1 - al[..., None]) * 255
    w = np.zeros(a.shape[:2], np.float32); w[band] = 0.7
    new = a * (1 - w[..., None]) + clean * w[..., None]; a = np.where((new - a) > 60, a + 60, new)
    Pm = prod & ~band; L = lum(a)
    if Pm.sum() < 100: return a8
    lo, hi = np.percentile(L[Pm], [0.7, 99.3])
    lo_t = min(lo * 0.5, 30.0) if lo > 8 else lo; hi_t = hi + (255 - hi) * 0.3 if hi < 250 else hi
    gain = min(253.0 / max(hi_t - lo_t, 1.0), 1.12)
    mx = a.max(2); mn = a.min(2); sat = (mx - mn) / (mx + 1e-3); neu = Pm & (sat < 0.12) & (L > 90) & (L < 235)
    wb = np.ones(3, np.float32)
    if neu.sum() > 800:
        m = a[neu].mean(0); g = m.mean()
        if np.abs(m - g).max() > 7: wb = np.clip(1 + 0.6 * (g / m - 1), 0.92, 1.08)
    t = np.clip((a * wb - lo_t) * gain, 0, 255); x = t / 255.0
    t = (0.88 * x + 0.12 * x * x * (3 - 2 * x)) * 255; t = np.minimum(t, 254.0)
    sharp = np.clip(t + 0.4 * (t - cv2.GaussianBlur(t, (0, 0), 1.2)), 0, 254)
    out = np.where(prod[..., None], np.where(Pm[..., None], sharp, t), 255.0)
    out = np.clip(np.rint(out), 0, 255).astype(np.uint8); out[bg] = 255
    return out
def enhance(cut, al):
    big = esr(cut)                                                     # 1600 x 1600, AI detail
    lan = np.asarray(Image.fromarray(cut).resize((1600, 1600), Image.LANCZOS), np.float32)
    mix = A.blend * big + (1 - A.blend) * lan                          # partial: keeps small print as photographed
    d = np.asarray(Image.fromarray(np.clip(mix, 0, 255).astype(np.uint8)).resize((800, 800), Image.LANCZOS), np.float32)
    a3 = np.clip(al, 0, 1)[..., None]
    d = d * a3 + 255 * (1 - a3)                                        # background stays pure white
    d[al < 0.02] = 255
    out = polish(np.clip(np.rint(d), 0, 255).astype(np.uint8)); out[al < 0.02] = 255
    return out

# ---------- downloading ----------
S = requests.Session(); S.headers["User-Agent"] = "CleanFoodFilter/0.1 (photo clean-up; contact via openfoodfacts app account)"
def get(u):
    for k in range(3):
        try:
            r = S.get(u, timeout=60)
            if r.status_code == 200 and len(r.content) > 500: return r.content
        except Exception: pass
        time.sleep(2 + 3 * k)
    raise RuntimeError("download failed")
def fetch(bc, u):
    try:
        try: data = get(u.replace(".400.jpg", ".full.jpg"))
        except Exception: data = get(u)
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB"); im.thumbnail((1200, 1200))
        return bc, np.asarray(im), None
    except Exception as e:
        return bc, None, repr(e)[:120]

# ---------- review sheets: original | cleaned+enhanced, 18 numbered tiles per sheet ----------
def sheets(items, name):
    out = []; T = 250
    for s in range(0, len(items), 18):
        ch = items[s:s + 18]; W = 3 * (2 * T + 30); H = 6 * (T + 28) + 10
        sh = Image.new("RGB", (W, H), (200, 200, 200)); dr = ImageDraw.Draw(sh); man = []
        for i, (bc, orig, enh) in enumerate(ch):
            x = (i % 3) * (2 * T + 30) + 10; y = (i // 3) * (T + 28) + 8
            o = Image.fromarray(orig); o.thumbnail((T, T)); bo = Image.new("RGB", (T, T), (235, 235, 235)); bo.paste(o, ((T - o.width) // 2, (T - o.height) // 2))
            sh.paste(bo, (x, y + 18)); sh.paste(Image.fromarray(enh).resize((T, T), Image.LANCZOS), (x + T + 6, y + 18))
            dr.text((x, y + 2), f"{s + i + 1}  {bc}", fill=(0, 0, 0)); man.append(bc)
        b = io.BytesIO(); sh.save(b, "PNG"); out.append((f"{name}_s{s // 18 + 1:02d}.png", b.getvalue(), man))
    return out

def run_shard(wave, si, items):
    tag = f"w{wave}_s{si:03d}"
    if f"shards/{tag}.zip" in have: print("skip", tag, flush=True); return
    t0 = time.time(); rows = []; review = []
    zb = io.BytesIO(); z = zipfile.ZipFile(zb, "w", zipfile.ZIP_STORED)
    with cf.ThreadPoolExecutor(8) as ex:
        for bc, rgb, err in ex.map(lambda kv: fetch(*kv), items):
            t = time.time()
            try:
                if err: raise RuntimeError(err)
                cut, al, notes, st = cutout(rgb)
                if cut is None: rows.append([bc, "none", 0, notes, "{}"]); continue
                enh = enhance(cut, al)
                for kind, arr in (("cut", cut), ("enh", enh)):
                    b = io.BytesIO(); Image.fromarray(arr).save(b, "WEBP", quality=90, method=5); z.writestr(f"{kind}/{bc}.webp", b.getvalue())
                rows.append([bc, "ok", round(time.time() - t, 2), notes, json.dumps(st)])
                o = rgb.copy() if max(rgb.shape[:2]) <= 400 else np.asarray(Image.fromarray(rgb).resize((max(1, rgb.shape[1] * 400 // max(rgb.shape[:2])), max(1, rgb.shape[0] * 400 // max(rgb.shape[:2])))))
                review.append((bc, o, enh))
            except Exception as e:
                rows.append([bc, "error", round(time.time() - t, 2), repr(e)[:160], "{}"])
    lb = io.StringIO(); csv.writer(lb).writerows(rows); z.writestr("log.csv", lb.getvalue())
    ss = sheets(review, tag)
    for fn, data, man in ss: z.writestr(f"sheets/{fn}", data)
    z.writestr("sheets/manifest.json", json.dumps({fn: man for fn, _, man in ss}))
    z.close()
    api.upload_file(path_or_fileobj=zb.getvalue(), path_in_repo=f"shards/{tag}.zip", repo_id=A.repo, repo_type="dataset", commit_message=tag)
    for fn, data, _ in ss[:2]:   # first two sheets also as loose files for a quick look
        api.upload_file(path_or_fileobj=data, path_in_repo=f"peek/{fn}", repo_id=A.repo, repo_type="dataset", commit_message=tag + " peek")
    ok = sum(r[1] == "ok" for r in rows); dt = time.time() - t0
    print(f"{tag}: {ok}/{len(rows)} ok, {sum(r[1]=='error' for r in rows)} errors, {dt:.0f}s ({dt/max(len(rows),1):.2f}s/photo)", flush=True)

T0 = time.time()
only = {int(x) for x in A.only_shards.split(",") if x != ""}
for w in A.waves.split(","):
    urls = requests.get(f"{A.src}/wave_{w}.json", timeout=60).json()
    items = sorted(urls.items())
    if A.limit: items = items[:A.limit]
    for si in range(0, len(items), A.shard):
        if only and si // A.shard not in only: continue
        run_shard(w, si // A.shard, items[si:si + A.shard])
print(f"ALL DONE in {time.time() - T0:.0f}s", flush=True)
