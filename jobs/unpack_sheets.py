# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub"]
# ///
"""Copies the review sheets out of each finished shard zip into loose files (sheets/<shard>/...), so reviewers can open
them one at a time. Waits for new shards until --expect shards are unpacked or --hours run out. Cheap CPU job."""
import argparse, io, os, tempfile, time, zipfile
from huggingface_hub import HfApi, hf_hub_download
P = argparse.ArgumentParser()
P.add_argument("--repo", default="DrNimbus/cff-photo-ai")
P.add_argument("--expect", type=int, default=111)
P.add_argument("--hours", type=float, default=11.5)
A = P.parse_args(); api = HfApi(); end = time.time() + A.hours * 3600
while True:
    files = set(api.list_repo_files(A.repo, repo_type="dataset"))
    shards = sorted(f for f in files if f.startswith("shards/") and f.endswith(".zip"))
    done = {f.split("/")[1] for f in files if f.startswith("sheets/") and f.endswith("/DONE")}
    todo = [s for s in shards if s[7:-4] not in done]
    for s in todo:
        tag = s[7:-4]
        with tempfile.TemporaryDirectory() as d:
            z = zipfile.ZipFile(hf_hub_download(A.repo, s, repo_type="dataset", cache_dir=d))
            out = os.path.join(d, "out"); os.makedirs(out)
            for n in z.namelist():
                if n.startswith("sheets/") or n == "log.csv":
                    open(os.path.join(out, os.path.basename(n)), "wb").write(z.read(n))
            open(os.path.join(out, "DONE"), "w").write("ok\n")
            api.upload_folder(folder_path=out, path_in_repo=f"sheets/{tag}", repo_id=A.repo, repo_type="dataset", commit_message=f"sheets {tag}")
        print("unpacked", tag, flush=True)
    n = len(done) + len(todo)
    if n >= A.expect or time.time() > end: print("finished", n, flush=True); break
    time.sleep(120)
