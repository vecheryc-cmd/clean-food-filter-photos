# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch", "torchvision", "transformers>=4.51,<4.58", "accelerate", "qwen-vl-utils", "pillow", "numpy", "huggingface_hub",
# ]
# ///
"""Automatic first-pass review of the cleaned photos with an open vision model (Qwen2.5-VL-7B-Instruct, Apache 2.0).
For each photo: original (tile cut from the review sheet) + cleaned version (from the shard zip) -> four yes/no checks.
Writes vlm/<shard>.csv (tile, barcode, verdict, missing, extra, ragged, not_product, reason). Skips shards already done."""
import argparse, csv, io, json, re, time, zipfile
import torch
from PIL import Image
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
P = argparse.ArgumentParser()
P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
P.add_argument("--shards", default="wt120_s000")          # comma list, or "all"
P.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
P.add_argument("--hours", type=float, default=0)            # >0: keep polling for new shards this long
A = P.parse_args(); api = HfApi()
proc = AutoProcessor.from_pretrained(A.model, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(A.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
Q = ("Image 1 is a shopper's original photo of one grocery product. Image 2 is our cut-out of that product on a white "
     "background, meant for a catalogue. Compare them carefully and answer with JSON only, keys:\n"
     "missing: true if any part of the product visible in image 1 is missing or cut off in image 2 (top, lid, cap, corner, "
     "edge, a bite or hole in the pack), else false\n"
     "extra: true if image 2 shows anything that is not the product (hand, fingers, table, shelf, floor, another item, cable, "
     "background patches), else false\n"
     "ragged: true if the outline in image 2 is jagged, torn or has grey/coloured fringes, else false\n"
     "not_product: true if image 1 is not a photo of a grocery product (a screen, screenshot, receipt, person, shoe, empty shelf), else false\n"
     "reason: up to 6 words")
T = 250
def tile(sheet, i):
    x = (i % 3) * (2 * T + 30) + 10; y = (i // 3) * (T + 28) + 8 + 18
    return sheet.crop((x, y, x + T, y + T))
@torch.no_grad()
def judge(orig, cleaned):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": Q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[orig, cleaned], return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=80, do_sample=False)
    s = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"\{.*\}", s, re.S)
    try: d = json.loads(m.group(0)) if m else {}
    except Exception: d = {}
    f = lambda k: bool(d.get(k)) if isinstance(d.get(k), bool) else str(d.get(k, "")).lower() == "true"
    r = dict(missing=f("missing"), extra=f("extra"), ragged=f("ragged"), not_product=f("not_product"), reason=str(d.get("reason", s[:60]))[:80], parsed=bool(d))
    r["verdict"] = "wrong" if r["not_product"] else "bad" if (r["missing"] or r["extra"] or r["ragged"] or not r["parsed"]) else "ok"
    return r
def run(tag):
    if f"vlm/{tag}.csv" in set(api.list_repo_files(A.repo, repo_type="dataset")): print("skip", tag); return
    z = zipfile.ZipFile(hf_hub_download(A.repo, f"shards/{tag}.zip", repo_type="dataset"))
    man = json.loads(z.read("sheets/manifest.json")); rows = []; t0 = time.time(); n = 0
    for sf in sorted(man):
        sheet = Image.open(io.BytesIO(z.read(f"sheets/{sf}"))).convert("RGB")
        for i, bc in enumerate(man[sf]):
            cl = Image.open(io.BytesIO(z.read(f"enh/{bc}.webp"))).convert("RGB").resize((448, 448))
            r = judge(tile(sheet, i).resize((448, 448)), cl); n += 1
            rows.append([n, bc, r["verdict"], r["missing"], r["extra"], r["ragged"], r["not_product"], r["reason"]])
    b = io.StringIO(); w = csv.writer(b); w.writerow(["tile", "barcode", "verdict", "missing", "extra", "ragged", "not_product", "reason"]); w.writerows(rows)
    api.upload_file(path_or_fileobj=b.getvalue().encode(), path_in_repo=f"vlm/{tag}.csv", repo_id=A.repo, repo_type="dataset", commit_message=f"vlm {tag}")
    print(f"{tag}: {n} photos, {sum(r[2]=='ok' for r in rows)} ok, {time.time()-t0:.0f}s ({(time.time()-t0)/max(n,1):.2f}s each)", flush=True)
end = time.time() + A.hours * 3600
while True:
    files = set(api.list_repo_files(A.repo, repo_type="dataset"))
    tags = sorted(f[7:-4] for f in files if f.startswith("shards/") and f.endswith(".zip"))
    want = tags if A.shards == "all" else [t for t in A.shards.split(",") if t in tags]
    for t in want:
        if f"vlm/{t}.csv" not in files: run(t)
    if time.time() > end: break
    time.sleep(120)
print("ALL DONE", flush=True)
