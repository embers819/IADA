# IADA

**Imbalance-Aware Distributional Alignment of Heterogeneous Modalities for HER2 Status Prediction, MICCAI 2026** 


![IADA overview](imgs/overview.png)



## Pre-Requisites

```bash
conda create -n iada python=3.9
conda activate iada
pip install -r requirements.txt
```

## Prepare Your Data

CSV files are expected to contain three columns:

```text
mri_id,wsi_id,label
```

Recommended directory structure:

```text
DATA_ROOT_DIR/
  wsi_features/
    slide_001.pt
    slide_002.pt
    ...
  mri_volumes/
    case_001.h5
    case_002.h5
    ...
LABEL_DIR/
  HER2_internal.csv
  HER2_external.csv
```

## Internal 5-Fold Evaluation

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_iada.py \
  --mode internal_5fold \
  --internal_csv <LABEL_DIR>/HER2_internal.csv \
  --wsi_root <DATA_ROOT_DIR>/wsi_features \
  --mri_root <DATA_ROOT_DIR>/mri_volumes \
  --output_dir <OUTPUT_DIR>/her2_internal5
```

## External Evaluation

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python train_iada.py \
  --mode external \
  --internal_csv <LABEL_DIR>/HER2_internal.csv \
  --external_csv <LABEL_DIR>/HER2_external.csv \
  --wsi_root <DATA_ROOT_DIR>/wsi_features \
  --mri_root <DATA_ROOT_DIR>/mri_volumes \
  --output_dir <OUTPUT_DIR>/her2_external
```

You can also start from the template config:

```bash
python train_iada.py --config configs/her2_iada.yaml
```

## Citation

If you find this repository useful, please cite:



