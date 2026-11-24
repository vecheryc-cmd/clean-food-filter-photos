# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "requests", "huggingface_hub"]
# ///
"""QC sheets after the two-reviewer pass of the AI photo clean-up (2026-11-24).
1) eye-check: the ORIGINAL photo of every tile a reviewer marked "wrong" (barcode only, nutrition panel, shelf...),
   cropped from the review sheets, 24 per sheet at 250 px, numbered -> eyecheck/wrong_sNN.png + wrong_manifest.json
2) final look: a random sample of photos both reviewers passed, the finished photo (enh/) at 400 px, 9 per sheet
   -> eyecheck/final_sNN.png + final_manifest.json
Input: --list URL of a JSON {"wrong": [[shard, barcode], ...], "final": [[shard, barcode], ...]}."""
import argparse, io, json, zipfile, collections, requests
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
P = argparse.ArgumentParser(); P.add_argument("--list", required=True); P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
A = P.parse_args(); api = HfApi(); L = requests.get(A.list, timeout=60).json(); ops = []
T = 250
def add(path, data): ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data))
def png(im): b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()

# ---- 0) order check: our tile->barcode mapping (tile n = n-th barcode) must equal the sheets' own manifest order
bad = 0
for sh, codes in sorted(L.get("order", {}).items()):
    man = json.load(open(hf_hub_download(A.repo, f"sheets/{sh}/manifest.json", repo_type="dataset")))
    real = [bc for f in sorted(man) for bc in man[f]]
    if real != codes:
        bad += 1; k = next((i for i, (a, b) in enumerate(zip(real, codes)) if a != b), min(len(real), len(codes)))
        print("ORDER MISMATCH", sh, "len", len(real), len(codes), "first diff at tile", k + 1, flush=True)
print("order check:", len(L.get("order", {})), "shards,", bad, "mismatched", flush=True)

# ---- 1) wrong: crop the original (left half) of each tile from its review sheet
by = collections.defaultdict(list)
for sh, bc in L.get("wrong", []): by[sh].append(bc)
crops = []
for sh in sorted(by):
    man = json.load(open(hf_hub_download(A.repo, f"sheets/{sh}/manifest.json", repo_type="dataset")))
    where = {bc: (f, i) for f, lst in man.items() for i, bc in enumerate(lst)}
    cache = {}
    for bc in by[sh]:
        if bc not in where: print("not on a sheet:", sh, bc, flush=True); continue
        f, i = where[bc]
        if f not in cache: cache[f] = Image.open(hf_hub_download(A.repo, f"sheets/{sh}/{f}", repo_type="dataset")).convert("RGB")
        x = (i % 3) * (2 * T + 30) + 10; y = (i // 3) * (T + 28) + 8 + 18
        crops.append((sh, bc, cache[f].crop((x, y, x + T, y + T))))
wm = {}
for s in range(0, len(crops), 24):
    ch = crops[s:s + 24]; S = Image.new("RGB", (6 * (T + 10) + 10, 4 * (T + 26) + 10), (210, 210, 210)); d = ImageDraw.Draw(S); m = []
    for k, (sh, bc, im) in enumerate(ch):
        x = 10 + (k % 6) * (T + 10); y = 10 + (k // 6) * (T + 26)
        S.paste(im, (x, y + 16)); d.text((x, y + 2), f"{s + k + 1}  {bc}", fill=(0, 0, 0)); m.append([s + k + 1, sh, bc])
    name = f"wrong_s{s // 24 + 1:02d}.png"; add(f"eyecheck/{name}", png(S)); wm[name] = m
add("eyecheck/wrong_manifest.json", json.dumps(wm).encode())
print("wrong tiles", len(crops), "sheets", len(wm), flush=True)

# ---- 2) final look: finished photo from the shard zip, large
by = collections.defaultdict(list)
for sh, bc in L.get("final", []): by[sh].append(bc)
ims = []
for sh in sorted(by):
    z = zipfile.ZipFile(hf_hub_download(A.repo, f"shards/{sh}.zip", repo_type="dataset"))
    for bc in by[sh]:
        try: ims.append((sh, bc, Image.open(io.BytesIO(z.read(f"enh/{bc}.webp"))).convert("RGB")))
        except KeyError: print("no enh:", sh, bc, flush=True)
F = 400; fm = {}
for s in range(0, len(ims), 9):
    ch = ims[s:s + 9]; S = Image.new("RGB", (3 * (F + 12) + 12, 3 * (F + 28) + 12), (190, 190, 190)); d = ImageDraw.Draw(S); m = []
    for k, (sh, bc, im) in enumerate(ch):
        x = 12 + (k % 3) * (F + 12); y = 12 + (k // 3) * (F + 28)
        S.paste(im.resize((F, F), Image.LANCZOS), (x, y + 18)); d.text((x, y + 2), f"{s + k + 1}  {bc}", fill=(0, 0, 0)); m.append([s + k + 1, sh, bc])
    name = f"final_s{s // 9 + 1:02d}.png"; add(f"eyecheck/{name}", png(S)); fm[name] = m
add("eyecheck/final_manifest.json", json.dumps(fm).encode())
print("final photos", len(ims), "sheets", len(fm), flush=True)
for i in range(0, len(ops), 200):
    api.create_commit(A.repo, repo_type="dataset", operations=ops[i:i + 200], commit_message="QC sheets: eye-check of wrong + final look sample")
print("ALL DONE", flush=True)
