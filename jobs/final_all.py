# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "requests", "huggingface_hub"]
# ///
"""Final look for every photo both reviewers passed (2026-11-24): the sample showed ~35% still fail a strict
catalogue look (pack cut by the photo frame, missing caps, ragged bag edges), so every passed photo gets one.
Finished photo (enh/) at 400 px, 9 per sheet, JPEG q88 -> finalall/fl_NNNN.jpg + finalall/manifest.json.
Input: --list URL of a JSON {"final": [[shard, barcode], ...]}. Sheets keep the list order (grouped by shard)."""
import argparse, io, json, zipfile, collections, os, requests
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
P = argparse.ArgumentParser(); P.add_argument("--list", required=True); P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
A = P.parse_args(); api = HfApi(); L = requests.get(A.list, timeout=60).json()["final"]
by = collections.OrderedDict()
for sh, bc in L: by.setdefault(sh, []).append(bc)
F = 400; ops = []; man = {}; buf = []; n = 0
def flush(final=False):
    global ops
    if ops and (final or len(ops) >= 150):
        api.create_commit(A.repo, repo_type="dataset", operations=ops, commit_message=f"final look sheets up to {n}"); ops = []
def sheet(items):
    global n
    n += 1; S = Image.new("RGB", (3 * (F + 12) + 12, 3 * (F + 28) + 12), (190, 190, 190)); d = ImageDraw.Draw(S); m = []
    for k, (sh, bc, im) in enumerate(items):
        x = 12 + (k % 3) * (F + 12); y = 12 + (k // 3) * (F + 28); no = (n - 1) * 9 + k + 1
        S.paste(im.resize((F, F), Image.LANCZOS), (x, y + 18)); d.text((x, y + 2), f"{no}  {bc}", fill=(0, 0, 0)); m.append([no, sh, bc])
    b = io.BytesIO(); S.save(b, "JPEG", quality=88); name = f"fl_{n:04d}.jpg"
    ops.append(CommitOperationAdd(path_in_repo=f"finalall/{name}", path_or_fileobj=b.getvalue())); man[name] = m; flush()
for sh, codes in by.items():
    p = hf_hub_download(A.repo, f"shards/{sh}.zip", repo_type="dataset"); z = zipfile.ZipFile(p)
    for bc in codes:
        try: buf.append((sh, bc, Image.open(io.BytesIO(z.read(f"enh/{bc}.webp"))).convert("RGB")))
        except KeyError: print("no enh:", sh, bc, flush=True)
        if len(buf) == 9: sheet(buf); buf = []
    z.close(); os.remove(os.path.realpath(p))
    print("done", sh, "sheets", n, flush=True)
if buf: sheet(buf)
ops.append(CommitOperationAdd(path_in_repo="finalall/manifest.json", path_or_fileobj=json.dumps(man).encode())); flush(True)
print("ALL DONE sheets", n, flush=True)
