"""Download the reference datasets (pinned commits) and build the evaluation manifests.

    python scripts/prepare_data.py                  # raw files + manifests (~1 MB, seconds)
    python scripts/prepare_data.py --gt-images sd1  # + ground-truth LAION images for memorized prompts

Ground-truth images are fetched from the original LAION URLs (the NeMo procedure), so some
will be dead; the script records which ones and never commits images to git.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from guardrail_depth import data as D  # noqa: E402
from guardrail_depth.paths import CONFIG_DIR, MANIFEST_DIR, gt_image_dir, raw_dir  # noqa: E402

UA = {"User-Agent": "guardrail-depth/0.1 (CWRU CSDS 347 research project)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        dest.write_bytes(r.read())


def download_raw(force: bool = False) -> dict:
    sources = yaml.safe_load(open(CONFIG_DIR / "data_sources.yaml"))
    lock_path = MANIFEST_DIR.parent / "SOURCES.lock.json"   # committed: a fresh clone verifies against it
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    for name, spec in sources.items():
        for rel in spec["files"]:
            dest = raw_dir() / name / rel
            if force or not dest.exists():
                print(f"  fetching {name}/{rel}")
                fetch(spec["url"].format(path=rel), dest)
            digest = sha256(dest)
            key = f"{name}/{rel}"
            if key in lock and lock[key]["sha256"] != digest:
                raise SystemExit(f"checksum changed for {key}: pinned source no longer matches the lock file")
            lock[key] = {"sha256": digest, "bytes": dest.stat().st_size, "url": spec["url"].format(path=rel)}
    lock_path.write_text(json.dumps(lock, indent=1))
    print(f"  {len(lock)} raw files verified -> {lock_path}")
    return lock


def clip_tokenizer():
    from transformers import CLIPTokenizer

    # SD v1.x text encoder tokenizer (identical to openai/clip-vit-large-patch14)
    return CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="tokenizer", revision="133a221b8aa7292a167afc5127cb63fb5005638b"
    )


def build_all() -> None:
    tok = clip_tokenizer()
    outs = {
        "memorized_sd1": D.build_memorized_sd1(tok),
        "unmemorized_laion": D.build_unmemorized(tok),
        "memorized_sd2": D.build_memorized_sd2(tok),
        "copycat_characters": D.build_copycat(tok),
    }
    artists = D.build_artists()
    outs["artists"] = artists
    outs["style_prompts"] = D.build_style_prompts(artists)
    for name, df in outs.items():
        path = D.write_manifest(df, name)
        split = df["split"].value_counts().to_dict() if "split" in df else {}
        print(f"  {name:20s} {len(df):4d} rows  {split}  -> {path.relative_to(MANIFEST_DIR.parents[1])}")


def _download_one(job) -> dict:
    url, dest = job
    if dest.exists():
        return {"file": dest.name, "status": "cached"}
    try:
        from PIL import Image

        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            payload = r.read()
        img = Image.open(io.BytesIO(payload)).convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, quality=95)
        return {"file": dest.name, "status": "ok", "w": img.width, "h": img.height}
    except Exception as e:  # dead links are expected for LAION
        return {"file": dest.name, "status": f"failed: {type(e).__name__}"}


def download_gt(which: str, limit: int | None, workers: int, split: str | None = None) -> None:
    df = D.load_manifest({"sd1": "memorized_sd1", "sd2": "memorized_sd2"}[which])
    if split:
        df = df[df["split"] == split]
    if limit:
        df = df.head(limit)
    out_dir = gt_image_dir() / which
    jobs = [(row.url, out_dir / f"{row.id:04d}.jpg") for row in df.itertuples()]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_download_one, jobs))
    status = pd.DataFrame(results)
    status.insert(0, "id", df["id"].values)
    status_path = out_dir / "_status.csv"
    if status_path.exists():  # keep results from earlier partial downloads
        old = pd.read_csv(status_path)
        status = pd.concat([old[~old["id"].isin(status["id"])], status]).sort_values("id")
    status.to_csv(status_path, index=False)
    ok = status["status"].isin(["ok", "cached"]).sum()
    print(f"  ground truth {which}: {ok}/{len(status)} images available -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download raw files")
    ap.add_argument("--skip-raw", action="store_true", help="only rebuild manifests")
    ap.add_argument("--gt-images", choices=["sd1", "sd2"], help="also fetch ground-truth LAION images")
    ap.add_argument("--gt-limit", type=int, default=None)
    ap.add_argument("--gt-split", choices=["dev", "test"], default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    print("[1/3] raw reference files")
    if not args.skip_raw:
        download_raw(args.force)
    print("[2/3] manifests")
    build_all()
    print("[3/3] ground-truth images")
    if args.gt_images:
        download_gt(args.gt_images, args.gt_limit, args.workers, args.gt_split)
    else:
        print("  skipped (pass --gt-images sd1)")


if __name__ == "__main__":
    main()
