# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "huggingface_hub", "pillow", "numpy"]
# ///
"""(v2, 2026-11-24: Colin: the photos look great but sit small in the white space. Each photo is re-framed: cropped to
the product (any pixel below pure white) and placed on an 800x800 white square filling FILL of the longer side, to match
the earlier cleaned photos, which fill the frame. Overwrites final/; nothing else changes.)
Publish the AI photos that passed the final look (2026-11-24): copy enh/<code>.webp from each shard zip to
final/<first 3 digits>/<code>.webp in the same PRIVATE dataset, where the Vercel route (HF_TOKEN) reads them for R2.
Input: --list URL of JSON {"items": [[shard, barcode], ...]}. Skips files already in final/."""
import argparse, collections, io, os, zipfile, requests
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
P = argparse.ArgumentParser(); P.add_argument("--list", required=True); P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
P.add_argument("--fill", type=float, default=0.96); P.add_argument("--overwrite", action="store_true")
A = P.parse_args(); api = HfApi()
L = requests.get(A.list, timeout=60).json()["items"]
have = {f for f in api.list_repo_files(A.repo, repo_type="dataset") if f.startswith("final/")}
by = collections.OrderedDict()
for sh, bc in L: by.setdefault(sh, []).append(bc)
ops = []; done = skipped = missing = 0
def reframe(data):
    im = Image.open(io.BytesIO(data)).convert("RGB"); a = np.asarray(im)
    m = a.min(2) < 248                                     # white background (lossy WebP leaves faint noise near edges)
    rows = np.where(m.sum(1) > 2)[0]; cols = np.where(m.sum(0) > 2)[0]
    if len(rows) == 0 or len(cols) == 0: return None
    y0, y1 = max(rows[0] - 2, 0), min(rows[-1] + 3, 800); x0, x1 = max(cols[0] - 2, 0), min(cols[-1] + 3, 800)
    c = im.crop((x0, y0, x1, y1)); w, h = c.size
    k = 800 * A.fill / max(w, h); c = c.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
    out = Image.new("RGB", (800, 800), (255, 255, 255)); out.paste(c, ((800 - c.width) // 2, (800 - c.height) // 2))
    b = io.BytesIO(); out.save(b, "WEBP", quality=90, method=6); return b.getvalue()
def flush(force=False):
    global ops
    if ops and (force or len(ops) >= 400):
        api.create_commit(A.repo, repo_type="dataset", operations=ops, commit_message=f"final AI photos ({done} so far)"); ops = []
for sh, codes in by.items():
    p = hf_hub_download(A.repo, f"shards/{sh}.zip", repo_type="dataset"); z = zipfile.ZipFile(p)
    for bc in codes:
        path = f"final/{bc[:3]}/{bc}.webp"
        if path in have and not A.overwrite: skipped += 1; continue
        try: data = z.read(f"enh/{bc}.webp")
        except KeyError: missing += 1; print("no enh:", sh, bc, flush=True); continue
        if data[:4] != b"RIFF" or data[8:12] != b"WEBP": missing += 1; print("not webp:", sh, bc, flush=True); continue
        data = reframe(data)
        if data is None: missing += 1; print("empty:", sh, bc, flush=True); continue
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)); done += 1; flush()
    z.close(); os.remove(os.path.realpath(p)); print("done", sh, "copied", done, flush=True)
flush(True)
print("ALL DONE copied", done, "skipped", skipped, "missing", missing, flush=True)
