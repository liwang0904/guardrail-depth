"""Print the environment: package versions, accelerator, memory, storage locations.

    python scripts/check_env.py            # report
    python scripts/check_env.py --strict   # exit 1 if a pinned version differs
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def pinned() -> dict[str, str]:
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip()] = ver.strip()
    return pins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    print(f"python      {sys.version.split()[0]} ({sys.executable})")
    print(f"platform    {platform.platform()}")
    mismatches = []
    for name, want in pinned().items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = "MISSING"
        ok = have.split("+")[0] == want
        if not ok:
            mismatches.append((name, want, have))
        print(f"  {'ok ' if ok else '!! '} {name:18s} {have:14s} (pinned {want})")

    import torch

    print(f"torch       {torch.__version__}  cuda build={torch.version.cuda}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  cuda:{i}    {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}  "
                  f"bf16={torch.cuda.is_bf16_supported()}")
        device = "cuda"
    elif torch.backends.mps.is_available():
        print("  mps       Apple GPU available (fp16)")
        device = "mps"
    else:
        print("  cpu only  (fine for tests; experiments need a GPU)")
        device = "cpu"
    x = torch.randn(256, 256, device=device)
    print(f"  matmul on {device}: ok ({float((x @ x).abs().mean()):.2f})")

    from guardrail_depth import paths

    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))
    for label, p in [("HF_HOME", Path(hf_home)), ("GD_DATA_ROOT", paths.data_root()),
                     ("GD_OUTPUT_ROOT", paths.output_root())]:
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free / 2**30
        print(f"{label:14s} {p}  ({free:.0f} GiB free)")
    if Path.home().as_posix().startswith("/home") and str(Path(hf_home)).startswith(str(Path.home())):
        print("  !! HF_HOME is inside your home directory; on Markov point it at course storage (25 GB quota)")
    if mismatches:
        print(f"{len(mismatches)} package(s) differ from requirements.txt")
        return 1 if args.strict else 0
    print("environment matches requirements.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
