# Food-type stand-in pictures (illus/)

These pictures are **generated stand-ins, not product photos**. Each shows a generic kind of food (for example
"Sliced bread loaf" or "Salsa"). The Clean Food Filter app shows one only for a product that has no usable photo of
its own, and labels it "Illustration, not this product". The product's own name, brand, ingredients and data are
unchanged.

- Generated with **Z-Image-Turbo** by Tongyi-MAI (https://huggingface.co/Tongyi-MAI/Z-Image-Turbo).
  - Licence: Apache License 2.0 (model card front matter `license: apache-2.0`). The licence allows commercial use
    and places no restriction on the outputs.
  - Settings: 9 scheduler steps (8 model passes), guidance 0, scheduler shift 3, 1024 x 1024, fixed seeds.
- Generated on two Hugging Face Jobs GPU runs (24 November 2026). Job script: `jobs/illus_gen.py`; prompts:
  `jobs/illus_types.json`.
- Finished in the Clean Food Filter workspace (`scripts/illus/publish_illus.py`): background lifted to pure white,
  cropped to the item, centred, 800 x 800 WebP.
- No brand, logo, label text or real package is shown. Every picture was looked at before use.
