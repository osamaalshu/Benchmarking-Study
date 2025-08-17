#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

from _shared import resolve_upstream_repo_path, default_results_dirs


def main():
    parser = argparse.ArgumentParser("Test pix2pix using upstream repo")
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--name", default="neurips_pix2pix")
    parser.add_argument("--direction", default="AtoB")
    parser.add_argument("--ext_repo", default=None, help="Path to pytorch-CycleGAN-and-pix2pix")
    parser.add_argument("--results_subdir", default="latest_test")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, help="Additional args to pass to test.py")
    args = parser.parse_args()

    ext_repo = resolve_upstream_repo_path(args.ext_repo)
    _, results = default_results_dirs(args.name)

    cmd = [
        "python", "test.py",
        "--model", "pix2pix",
        "--dataset_mode", "aligned",
        "--dataroot", str(Path(args.dataroot).resolve()),
        "--name", args.name,
        "--direction", args.direction,
        "--results_dir", str((results / args.results_subdir).resolve()),
    ]
    if args.extra_args:
        cmd += args.extra_args

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ext_repo))


if __name__ == "__main__":
    main()


