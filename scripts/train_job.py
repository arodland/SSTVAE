# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.2",
#     "torchvision",
#     "numpy",
#     "scipy",
#     "pillow",
#     "lpips",
#     "datasets",
#     "huggingface_hub",
#     "matplotlib",
# ]
# ///
"""HF Jobs / remote-machine entry point for SSTVAE stage-1 training.

Pulls the project code from a Hub model repo (uploaded by
scripts/launch_job.sh), then runs scripts/train.py with Hub dataset
in and Hub checkpoints out. All configuration via environment:

    SSTVAE_CODE_REPO   code snapshot repo   (default arodland/sstvae-code)
    SSTVAE_DATASET     dataset repo         (default arodland/coco640-sstvae)
    SSTVAE_OUT_REPO    checkpoint repo      (default arodland/sstvae-s1-640)
    SSTVAE_ARGS        extra train.py args  (e.g. "--epochs 60 --batch 64")
    SSTVAE_SCRIPT      training script in scripts/ (default train.py;
                       train_refiner.py for the post-decoder refiner)
    SSTVAE_RESUME      set to resume from SSTVAE_OUT_REPO's checkpoint

Run on HF Jobs:
    hf jobs uv run scripts/train_job.py --flavor l4x1 --timeout 12h \
        --secrets HF_TOKEN --env SSTVAE_ARGS="--epochs 60 --batch 48"

Run on any GPU box with uv:
    HF_TOKEN=... uv run scripts/train_job.py
"""

import os
import runpy
import sys

from huggingface_hub import snapshot_download

code_repo = os.environ.get("SSTVAE_CODE_REPO", "arodland/sstvae-code")
dataset = os.environ.get("SSTVAE_DATASET", "arodland/coco640-sstvae")
out_repo = os.environ.get("SSTVAE_OUT_REPO", "arodland/sstvae-s1-640")
extra = os.environ.get("SSTVAE_ARGS", "--epochs 60 --batch 48").split()
script = os.environ.get("SSTVAE_SCRIPT", "train.py")

code_dir = snapshot_download(code_repo)
sys.path.insert(0, code_dir)

argv = [
    script,
    "--hf-dataset", dataset,
    "--push-to-hub", out_repo,
    "--out", "out",
    *extra,
]
if os.environ.get("SSTVAE_RESUME"):
    argv += ["--resume", f"hf://{out_repo}"]

print(f"code={code_repo} dataset={dataset} out={out_repo} argv={argv[1:]}")
sys.argv = argv
runpy.run_path(os.path.join(code_dir, "scripts", script), run_name="__main__")
