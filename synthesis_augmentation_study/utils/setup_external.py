#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"


def run(cmd, cwd=None):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser("Setup upstream pytorch-CycleGAN-and-pix2pix repo")
    parser.add_argument("--clone", action="store_true", help="Clone the upstream repo into synthesis/external/")
    parser.add_argument("--branch", default="master", help="Branch or tag to checkout")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ext_dir = root / "synthesis" / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)

    if args.clone:
        if any(ext_dir.iterdir()):
            print(f"External dir not empty: {ext_dir}. Skipping clone.")
        else:
            run(["git", "clone", REPO_URL, str(ext_dir)])
            run(["git", "checkout", args.branch], cwd=str(ext_dir))

    print("External repo ready at:", ext_dir)


if __name__ == "__main__":
    main()


