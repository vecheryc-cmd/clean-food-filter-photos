# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "diffusers>=0.36", "transformers>=4.51", "accelerate", "pillow", "requests", "huggingface_hub", "sentencepiece"]
# ///
"""Pilot for Colin's photo-Q1..Q3 answers (2026-11-24), both models Apache 2.0:
1) food-kind illustrations (Qwen-Image): unbranded packs, no text, two styles, for products with no photo;
2) AI edit (Qwen-Image-Edit-2509) of real photos: remove the hand / rebuild a poor photo, keep the pack's print.
Outputs to the private dataset under gen_pilot/ plus contact sheets."""
import io, json, time, requests, torch
from PIL import Image, ImageDraw
from huggingface_hub import HfApi
REPO = "DrNimbus/cff-photo-ai"; api = HfApi()
INP = requests.get("https://raw.githubusercontent.com/vecheryc-cmd/clean-food-filter-photos/main/jobs/gen_pilot_inputs.json", timeout=60).json()
def up(img, path):
    b = io.BytesIO(); img.save(b, "PNG"); api.upload_file(path_or_fileobj=b.getvalue(), path_in_repo=path, repo_id=REPO, repo_type="dataset", commit_message=path)
def sheet(pairs, cols, T, path):
    rows = (len(pairs) + cols - 1) // cols; S = Image.new("RGB", (cols * (T + 10) + 10, rows * (T + 30) + 10), (210, 210, 210)); d = ImageDraw.Draw(S)
    for i, (label, im) in enumerate(pairs):
        x = 10 + (i % cols) * (T + 10); y = 10 + (i // cols) * (T + 30); im = im.copy(); im.thumbnail((T, T))
        S.paste(Image.new("RGB", (T, T), (255, 255, 255)), (x, y + 20)); S.paste(im, (x + (T - im.width) // 2, y + 20 + (T - im.height) // 2)); d.text((x, y + 4), label, fill=(0, 0, 0))
    up(S, path)
t0 = time.time()
# ---------- 1) illustrations ----------
from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16).to("cuda")
KINDS = ["a glass jar of tomato salsa", "a carton of a dozen brown eggs, lid open", "a plastic gallon jug of milk", "a jar of peanut butter",
         "a bag of sliced sandwich bread", "a bag of tortilla chips"]
STYLES = {"photo": "Professional e-commerce studio product photo of {k}. Plain unbranded packaging with no text, no logo, no letters, no numbers, blank label. Centered, front view, soft even lighting, pure white background, high detail.",
          "illus": "Clean modern 3D illustration of {k}, soft shading, gentle colours, plain unbranded packaging with no text, no logo, no letters, no numbers. Centered on a pure white background, catalogue icon style."}
NEG = "text, words, letters, logo, brand, watermark, label text, numbers, hands, people, background clutter"
out = []
for s, tmpl in STYLES.items():
    for i, k in enumerate(KINDS):
        im = pipe(prompt=tmpl.format(k=k), negative_prompt=NEG, width=1024, height=1024, num_inference_steps=40, true_cfg_scale=4.0,
                  generator=torch.Generator("cuda").manual_seed(7 + i)).images[0]
        up(im, f"gen_pilot/illus/{s}_{i}.png"); out.append((f"{s}: {k[:34]}", im))
sheet(out, 4, 300, "gen_pilot/sheet_illustrations.png")
print("illustrations done", round(time.time() - t0), "s", flush=True)
del pipe; torch.cuda.empty_cache()
# ---------- 2) edits of real photos ----------
from diffusers import QwenImageEditPlusPipeline
ed = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509", torch_dtype=torch.bfloat16).to("cuda")
P_HAND = ("Remove the hand and fingers holding the product and show the complete product on its own, front view, on a pure white "
          "background, like a professional catalogue photo. Keep every printed word, logo, number and colour on the package exactly as it is. "
          "Do not add or change any text.")
P_POOR = ("Turn this into a professional catalogue photo of the same product: straighten it, even lighting, sharp focus, pure white background. "
          "Keep every printed word, logo, number and colour on the package exactly as it is. Do not add or change any text.")
pairs = []
for kind, prompt in (("hands", P_HAND), ("poor", P_POOR)):
    for bc, u in INP[kind].items():
        try:
            r = requests.get(u.replace(".400.jpg", ".full.jpg"), timeout=60, headers={"User-Agent": "CleanFoodFilter/0.1"})
            if r.status_code != 200: r = requests.get(u, timeout=60)
            src = Image.open(io.BytesIO(r.content)).convert("RGB"); src.thumbnail((1024, 1024))
            im = ed(image=[src], prompt=prompt, negative_prompt=" ", num_inference_steps=40, true_cfg_scale=4.0,
                    generator=torch.Generator("cuda").manual_seed(1)).images[0]
            up(im, f"gen_pilot/edit/{kind}_{bc}.png"); pairs += [(f"{bc} original", src), (f"{bc} AI edit ({kind})", im)]
        except Exception as e:
            print("edit failed", bc, repr(e)[:200], flush=True)
for j in range(0, len(pairs), 12): sheet(pairs[j:j + 12], 4, 380, f"gen_pilot/sheet_edits_{j // 12 + 1}.png")
print("ALL DONE", round(time.time() - t0), "s", flush=True)
