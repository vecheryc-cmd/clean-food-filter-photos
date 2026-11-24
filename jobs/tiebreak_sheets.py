# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "requests", "huggingface_hub"]
# ///
"""Tie-break sheets for the AI photo final look (2026-11-24): photos where the first reviewer failed and a second,
independent reviewer passed. Finished photo (enh/) at 400 px, 9 per sheet, labelled with the ORIGINAL photo number,
JPEG q88 -> <out>/tb_NNNN.jpg + <out>/manifest.json. Input: --list URL of JSON {"items": [[number, shard, barcode], ...]}."""
import argparse, io, json, zipfile, collections, os, requests
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
P = argparse.ArgumentParser(); P.add_argument("--list", required=True); P.add_argument("--out", default="tiebreak")
P.add_argument("--repo", default="DrNimbus/cff-photo-ai"); A = P.parse_args(); api = HfApi()
L = requests.get(A.list, timeout=60).json()["items"]
by = collections.OrderedDict()
for no, sh, bc in L: by.setdefault(sh, []).append((no, bc))
F = 400; ops = []; man = {}; buf = []; n = 0
def sheet(items):
    global n
    n += 1; S = Image.new("RGB", (3 * (F + 12) + 12, 3 * (F + 28) + 12), (190, 190, 190)); d = ImageDraw.Draw(S); m = []
    for k, (no, sh, bc, im) in enumerate(items):
        x = 12 + (k % 3) * (F + 12); y = 12 + (k // 3) * (F + 28)
        S.paste(im.resize((F, F), Image.LANCZOS), (x, y + 18)); d.text((x, y + 2), f"{no}  {bc}", fill=(0, 0, 0)); m.append([no, sh, bc])
    b = io.BytesIO(); S.save(b, "JPEG", quality=88); name = f"tb_{n:04d}.jpg"
    ops.append(CommitOperationAdd(path_in_repo=f"{A.out}/{name}", path_or_fileobj=b.getvalue())); man[name] = m
for sh, items in by.items():
    p = hf_hub_download(A.repo, f"shards/{sh}.zip", repo_type="dataset"); z = zipfile.ZipFile(p)
    for no, bc in items:
        try: buf.append((no, sh, bc, Image.open(io.BytesIO(z.read(f"enh/{bc}.webp"))).convert("RGB")))
        except KeyError: print("no enh:", sh, bc, flush=True)
        if len(buf) == 9: sheet(buf); buf = []
    z.close(); os.remove(os.path.realpath(p)); print("done", sh, "sheets", n, flush=True)
if buf: sheet(buf)
ops.append(CommitOperationAdd(path_in_repo=f"{A.out}/manifest.json", path_or_fileobj=json.dumps(man).encode()))
for i in range(0, len(ops), 150):
    api.create_commit(A.repo, repo_type="dataset", operations=ops[i:i + 150], commit_message=f"tie-break sheets part {i // 150 + 1}")
print("ALL DONE sheets", n, flush=True)
