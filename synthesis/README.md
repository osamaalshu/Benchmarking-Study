## Data Synthesis Benchmarking

This module adds a self-contained workflow to benchmark Pix2Pix (paired; mask → image).

The implementation is from the upstream repository `pytorch-CycleGAN-and-pix2pix` [link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix?tab=readme-ov-file).

Nothing here modifies the existing training/evaluation code. All artifacts (datasets, checkpoints, results) live under `synthesis/`.

### Layout

- `synthesis/datasets/` prepared datasets in the exact folder format expected by the upstream repo
- `synthesis/checkpoints/` model checkpoints
- `synthesis/results/` generated images and HTML reports
- `synthesis/external/` upstream repository clone (optional; can point to any path)

### Requirements (isolated)

Install optional dependencies for HTML export/visualization used by the upstream repo:

```bash
pip install -r synthesis/requirements.txt
```

This is separate from the root `requirements.txt` to avoid affecting existing baselines.

### 1) Get the upstream code

Option A (recommended): clone into `synthesis/external/` once.

```bash
python synthesis/setup_external.py --clone
```

Option B: if you already have the repo elsewhere, pass `--ext_repo /path/to/pytorch-CycleGAN-and-pix2pix` to the train/test wrapper scripts below.

### 2) Prepare datasets

#### Pix2Pix (paired; A = masks, B = images)

Input expected: folders with `images/` and `labels/` (our preprocessed format), e.g. `./data/train-preprocessed`, `./data/val`, `./data/test`.

Creates aligned AB tiles where the left half is the colorized mask (A) and the right half is the microscopy image (B), as required by `dataset_mode=aligned`.

```bash
python synthesis/prepare_pix2pix_dataset.py \
  --train_path ./data/train-preprocessed \
  --val_path ./data/val \
  --test_path ./data/test \
  --dataset_name neurips_masks2imgs \
  --tile_size 256 --stride 256
```

Output: `synthesis/datasets/neurips_masks2imgs/{train,val,test}/*_AB.png`

### 3) Train and test wrappers

These are thin wrappers over the upstream `train.py`/`test.py` that write into `synthesis/checkpoints/` and `synthesis/results/`.

Pix2Pix:

```bash
python synthesis/train_pix2pix.py --dataroot synthesis/datasets/neurips_masks2imgs \
  --name neurips_pix2pix --direction AtoB

python synthesis/test_pix2pix.py --dataroot synthesis/datasets/neurips_masks2imgs \
  --name neurips_pix2pix --direction AtoB
```

Add `--ext_repo` if the upstream repo lives outside `synthesis/external/`.

### Notes

- For Pix2Pix labels, this pipeline colorizes 3-class masks to RGB so the generator can condition on color. Binary masks are supported (mapped to black/white).
- Tiling is configurable via `--tile_size` and `--stride` (no overlap by default).
- This module does not alter or import from the existing baseline code; it is fully standalone.
