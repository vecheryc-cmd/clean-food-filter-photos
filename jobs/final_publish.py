# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "huggingface_hub"]
# ///
"""Publish the AI photos that passed the final look (2026-11-24): copy enh/<code>.webp from each shard zip to
final/<first 3 digits>/<code>.webp in the same PRIVATE dataset, where the Vercel route (HF_TOKEN) reads them for R2.
Input: --list URL of JSON {"items": [[shard, barcode], ...]}. Skips files already in final/."""
import argparse, collections, os, zipfile, requests
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
P = argparse.ArgumentParser(); P.add_argument("--list", required=True); P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
A = P.parse_args(); api = HfApi()
L = requests.get(A.list, timeout=60).json()["items"]
have = {f for f in api.list_repo_files(A.repo, repo_type="dataset") if f.startswith("final/")}
by = collections.OrderedDict()
for sh, bc in L: by.setdefault(sh, []).append(bc)
ops = []; done = skipped = missing = 0
def flush(force=False):
    global ops
    if ops and (force or len(ops) >= 400):
        api.create_commit(A.repo, repo_type="dataset", operations=ops, commit_message=f"final AI photos ({done} so far)"); ops = []
for sh, codes in by.items():
    p = hf_hub_download(A.repo, f"shards/{sh}.zip", repo_type="dataset"); z = zipfile.ZipFile(p)
    for bc in codes:
        path = f"final/{bc[:3]}/{bc}.webp"
        if path in have: skipped += 1; continue
        try: data = z.read(f"enh/{bc}.webp")
        except KeyError: missing += 1; print("no enh:", sh, bc, flush=True); continue
        if data[:4] != b"RIFF" or data[8:12] != b"WEBP": missing += 1; print("not webp:", sh, bc, flush=True); continue
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data)); done += 1; flush()
    z.close(); os.remove(os.path.realpath(p)); print("done", sh, "copied", done, flush=True)
flush(True)
print("ALL DONE copied", done, "skipped", skipped, "missing", missing, flush=True)
