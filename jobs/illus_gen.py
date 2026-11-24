# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "diffusers==0.43.0", "transformers>=4.51", "accelerate", "pillow", "requests", "huggingface_hub", "sentencepiece"]
# ///
"""Clean Food Filter: food-type stand-in pictures ("Illustration, not this product") on one Hugging Face GPU job.

Model: Tongyi-MAI/Z-Image-Turbo (Apache-2.0), model-card settings: 9 scheduler steps (= 8 transformer passes),
guidance 0, scheduler shift 3 (the repo default), 1024 x 1024.
Memory: the 6B transformer (bf16 ~12 GB) and the Qwen3 text encoder (~8 GB) do not both fit next to activations on a
24 GB L4, so every prompt is encoded first with the text encoder on the GPU, the encoder goes back to the CPU, and
the transformer + VAE then generate from the stored embeddings.
Order: pass 1 = first seed for every type (most-used types first), pass 2 = second seed, so a time limit only ever
cuts second seeds.  Results go to the PRIVATE dataset (illus/<id>_s<seed>.webp, illus/log_<time>.csv) every --flush images;
a re-run skips images already there.  Contact sheets (12 types, both seeds) at the end: illus/sheets/sheet_NN.png.
"""
import argparse, io, os, time, csv
import requests, torch
from PIL import Image, ImageDraw
from huggingface_hub import HfApi, CommitOperationAdd, snapshot_download

T0 = time.time()
P = argparse.ArgumentParser()
P.add_argument("--types", default="https://raw.githubusercontent.com/vecheryc-cmd/clean-food-filter-photos/main/jobs/illus_types.json")
P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
P.add_argument("--seeds", default="11,27")
P.add_argument("--only", default="")              # comma list of type ids (re-run for chosen types)
P.add_argument("--deadline", type=float, default=50)  # minutes after start: no new image after this
P.add_argument("--flush", type=int, default=10)
P.add_argument("--sheets", type=int, default=1)
A = P.parse_args()
api = HfApi()
def log(*a): print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)

types = requests.get(A.types, timeout=60).json()["types"]
if A.only: types = [t for t in types if t["id"] in set(A.only.split(","))]
seeds = [int(s) for s in A.seeds.split(",")]
have = {f for f in api.list_repo_files(A.repo, repo_type="dataset") if f.startswith("illus/")}
log(len(types), "types,", len(have), "illus files already in the dataset")

path = snapshot_download("Tongyi-MAI/Z-Image-Turbo", allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer/*"])
log("model downloaded")
from diffusers import ZImagePipeline
pipe = ZImagePipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
log("pipeline loaded (CPU)")
pipe.text_encoder.to("cuda")
emb = {}
with torch.no_grad():
    for t in types:
        pe, _ = pipe.encode_prompt(prompt=t["prompt"], device="cuda", do_classifier_free_guidance=False)
        emb[t["id"]] = [e.to("cpu") for e in pe]
pipe.text_encoder.to("cpu"); torch.cuda.empty_cache()
log("prompts encoded:", len(emb))
pipe.transformer.to("cuda"); pipe.vae.to("cuda")
log("transformer on GPU; free memory GB:", round(torch.cuda.mem_get_info()[0] / 2**30, 1))

ops, rows, done, times = [], [], 0, []
LOGNAME = f"illus/log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
def flush(force=False):
    global ops
    if ops and (force or len(ops) >= A.flush):
        b = io.StringIO(); w = csv.writer(b); w.writerow(["id", "seed", "seconds"]); w.writerows(rows)
        ops.append(CommitOperationAdd(LOGNAME, b.getvalue().encode()))
        for k in range(3):
            try:
                api.create_commit(A.repo, repo_type="dataset", operations=ops, commit_message=f"illus: {done} images"); break
            except Exception as e:
                log("commit failed, retry", repr(e)[:200]); time.sleep(10)
        have.update(o.path_in_repo for o in ops); ops = []
imgs = {}
stop = False
for seed in seeds:
    for t in types:
        name = f"illus/{t['id']}_s{seed}.webp"
        if name in have: continue
        if (time.time() - T0) / 60 > A.deadline: stop = True; break
        s0 = time.time()
        with torch.no_grad():
            im = pipe(prompt_embeds=[e.to("cuda") for e in emb[t["id"]]], height=1024, width=1024, num_inference_steps=9,
                      guidance_scale=0.0, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        dt = time.time() - s0; times.append(dt); done += 1
        b = io.BytesIO(); im.save(b, "WEBP", quality=95, method=4)
        ops.append(CommitOperationAdd(name, b.getvalue())); rows.append([t["id"], seed, round(dt, 1)])
        im.thumbnail((300, 300)); imgs[(t["id"], seed)] = im
        if done <= 3 or done % 10 == 0:
            log(f"{done} images, last {dt:.1f} s, mean {sum(times) / len(times):.1f} s/image, GPU peak GB {torch.cuda.max_memory_allocated() / 2**30:.1f}")
        if len(ops) >= A.flush: flush()
    if stop: break
flush(True)
log("generation finished:", done, "new images;", "stopped at deadline" if stop else "all done")

if A.sheets:
    def thumb(tid, seed):
        if (tid, seed) in imgs: return imgs[(tid, seed)]
        f = f"illus/{tid}_s{seed}.webp"
        if f not in have: return None
        im = Image.open(api.hf_hub_download(A.repo, f, repo_type="dataset")).convert("RGB"); im.thumbnail((300, 300)); return im
    T = 300; per = 12; ops = []
    for n in range(0, len(types), per):
        if (time.time() - T0) / 60 > A.deadline + 5: log("no time left for more sheets"); break
        chunk = types[n:n + per]
        S = Image.new("RGB", (10 + len(seeds) * (T + 10) + 260, 10 + len(chunk) * (T + 10)), (200, 200, 200)); d = ImageDraw.Draw(S)
        for r, t in enumerate(chunk):
            y = 10 + r * (T + 10)
            d.text((10, y + 4), f"{n + r + 1}. {t['id']}", fill=(0, 0, 0))
            for c, seed in enumerate(seeds):
                im = thumb(t["id"], seed); x = 260 + c * (T + 10)
                if im is not None: S.paste(im, (x, y))
                d.text((x + 4, y + T - 14), f"seed {seed}", fill=(255, 0, 0))
        b = io.BytesIO(); S.save(b, "PNG"); ops.append(CommitOperationAdd(f"illus/sheets/sheet_{n // per + 1:02d}.png", b.getvalue()))
    if ops: api.create_commit(A.repo, repo_type="dataset", operations=ops, commit_message="illus: contact sheets")
    log("sheets written:", len(ops))
log("ALL DONE")
