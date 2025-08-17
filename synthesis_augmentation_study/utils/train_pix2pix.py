#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from _shared import resolve_upstream_repo_path, default_results_dirs


def main():
    parser = argparse.ArgumentParser("Train pix2pix using upstream repo")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--name", default="neurips_pix2pix")
    parser.add_argument("--direction", default="AtoB")
    parser.add_argument("--ext_repo", default=None, help="Path to pytorch-CycleGAN-and-pix2pix")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional args to pass to train.py")
    args = parser.parse_args()

    ext_repo = resolve_upstream_repo_path(args.ext_repo)
    checkpoints, _ = default_results_dirs(args.name)

    cmd = [
        "python", "train.py",
        "--model", "pix2pix",
        "--dataset_mode", "aligned",
        "--dataroot", str(Path(args.dataroot).resolve()),
        "--name", args.name,
        "--direction", args.direction,
        "--checkpoints_dir", str(checkpoints.resolve()),
    ]
    if args.extra_args:
        cmd += args.extra_args

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ext_repo))


if __name__ == "__main__":
    main()


