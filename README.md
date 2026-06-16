# IADA

Minimal release code for **Imbalance-Aware Distributional Alignment (IADA)**
for HER2 status prediction from paired WSI and MRI.

This repository keeps only the method components used by the paper:

- WSI encoder: RRT-MIL.
- MRI encoder: 3D DenseNet121.
- Distributional representation: modality-specific Gaussian mean and standard deviation heads.
- Distributional alignment: symmetric KL divergence plus standard normal KL regularization.
- Classification: sigmoid feature gates on modality means, concatenation, and focal-loss supervision.
- Modality rebalancing: Dynamic Gradient Modulation (DGM) from unimodal prediction scores.

The original development repository contains many comparison methods and
experimental utilities. They are intentionally not included here.

## Installation

```bash
conda activate nnMamba
cd /home/hxm/model/IADA_released
pip install -r requirements.txt
```

## Expected Data

CSV files are expected to contain three columns:

```text
mri_id,wsi_id,label
```

Headerless CSV is supported by default. Headered CSV is supported with
`--csv_has_header`.

Feature files can be addressed either by absolute paths in the CSV or by IDs
resolved under the supplied roots. The loader searches common suffixes:
`.pt`, `.pth`, `.h5`, `.hdf5`, `.npy`, and `.npz`.

Default HER2 paths:

- Internal labels: `/home/hxm/label/HER2/HER2_internal.csv`
- External labels: `/home/hxm/label/HER2/zssy_her2_label.csv`
- WSI root: `/home/hxm/data/HER2/WSI/512_R50_HER2_all/`
- MRI root: `/home/hxm/data/HER2/MRI/preprocess/MRI_64_padding_all_h5/`

## Internal 5-Fold Evaluation

```bash
python train_iada.py \
  --mode internal_5fold \
  --internal_csv /home/hxm/label/HER2/HER2_internal.csv \
  --wsi_root /home/hxm/data/HER2/WSI/512_R50_HER2_all/ \
  --mri_root /home/hxm/data/HER2/MRI/preprocess/MRI_64_padding_all_h5/ \
  --output_dir /home/hxm/model/IADA_released/runs/her2_internal5
```

## External Evaluation

Train on all internal data, then evaluate on the external CSV:

```bash
python train_iada.py \
  --mode external \
  --internal_csv /home/hxm/label/HER2/HER2_internal.csv \
  --external_csv /home/hxm/label/HER2/zssy_her2_label.csv \
  --wsi_root /home/hxm/data/HER2/WSI/512_R50_HER2_all/ \
  --mri_root /home/hxm/data/HER2/MRI/preprocess/MRI_64_padding_all_h5/ \
  --output_dir /home/hxm/model/IADA_released/runs/her2_external
```

## Real-Feature Smoke Test

Four anonymized real HER2 examples are provided under `examples/real_data/` on
the release server. They are for loader/training smoke tests only, not for
reporting model performance.

```bash
python train_iada.py --config configs/example_real_4cases.yaml
```

Confirm data-release permission before uploading these real derived files to a
public repository.

## Paper Defaults

The default hyperparameters match the method description:

- Optimizer: Adam.
- Learning rate: `1e-4`.
- Batch size: `2`.
- Epochs: `100`.
- Classification loss: Focal loss.
- `alpha = 0.2` for `(1 - alpha) * L_cls + alpha * L_dist`.
- `beta = 0.1` for DGM.
- `lambda_reg = 2e-2` for standard normal KL regularization.
- Gradient clipping is disabled by default.

Checkpoints, metrics, predictions, and logs are written below `--output_dir`.
