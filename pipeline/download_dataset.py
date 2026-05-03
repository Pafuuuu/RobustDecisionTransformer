import requests
import os

url = "http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/walker2d_medium_replay-v2.hdf5"
out = "datasets/walker2d-medium-replay-v2.hdf5"
os.makedirs("datasets", exist_ok=True)

print(f"Downloading {url} ...")
with requests.get(url, stream=True) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded/1e6:.1f} / {total/1e6:.1f} MB", end="", flush=True)
print(f"\nDone: {out}")
