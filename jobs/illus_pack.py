# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "huggingface_hub"]
# ///
"""Pack the generated food-type pictures (illus/<id>_s<seed>.webp in the PRIVATE dataset) into one zip and print a
short-lived signed download link ("ZIPURL ..."), so the review and the final crop can run in the workspace, which has
no Hugging Face login.  CPU only."""
import io, os, zipfile, requests
from huggingface_hub import HfApi, hf_hub_download
REPO = "DrNimbus/cff-photo-ai"; api = HfApi()
files = sorted(f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("illus/") and f.endswith(".webp"))
b = io.BytesIO()
with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as z:
    for f in files:
        z.write(hf_hub_download(REPO, f, repo_type="dataset"), f.split("/", 1)[1])
api.upload_file(path_or_fileobj=b.getvalue(), path_in_repo="illus/pack/all_raw.zip", repo_id=REPO, repo_type="dataset",
                commit_message=f"illus: pack {len(files)} pictures")
r = requests.head(f"https://huggingface.co/datasets/{REPO}/resolve/main/illus/pack/all_raw.zip",
                  headers={"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}, allow_redirects=False)
print("FILES", len(files), "ZIPBYTES", len(b.getvalue()), flush=True)
print("ZIPURL", r.headers.get("Location"), flush=True)
