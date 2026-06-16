import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_KEYS = ("features", "feature", "feats", "embeddings", "embedding", "x", "data")
VOLUME_KEYS = ("image", "volume", "mri", "data", "features")


def read_label_csv(csv_path, has_header=False, mri_col=0, wsi_col=1, label_col=2):
    if has_header:
        df = pd.read_csv(csv_path)
        if isinstance(mri_col, int):
            mri_col = df.columns[mri_col]
        if isinstance(wsi_col, int):
            wsi_col = df.columns[wsi_col]
        if isinstance(label_col, int):
            label_col = df.columns[label_col]
        out = df[[mri_col, wsi_col, label_col]].copy()
    else:
        df = pd.read_csv(csv_path, header=None)
        out = df.iloc[:, [mri_col, wsi_col, label_col]].copy()
    out.columns = ["mri_id", "wsi_id", "label"]
    out["mri_id"] = out["mri_id"].astype(str)
    out["wsi_id"] = out["wsi_id"].astype(str)
    out["label"] = out["label"].astype(int)
    return out


def _resolve_file(root, item):
    path = Path(item)
    if path.is_absolute() and path.exists():
        return str(path)

    root_path = Path(root)
    direct = root_path / item
    if direct.exists():
        return str(direct)

    stem = str(item)
    suffixes = ("", ".pt", ".pth", ".h5", ".hdf5", ".npy", ".npz")
    for suffix in suffixes:
        candidate = root_path / f"{stem}{suffix}"
        if candidate.exists():
            return str(candidate)

    matches = []
    for suffix in ("*.pt", "*.pth", "*.h5", "*.hdf5", "*.npy", "*.npz"):
        matches.extend(glob.glob(str(root_path / f"{stem}*{suffix[1:]}")))
    if matches:
        return sorted(matches)[0]

    raise FileNotFoundError(f"Could not resolve '{item}' under '{root}'.")


def _first_h5_dataset(handle, preferred_keys):
    for key in preferred_keys:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return handle[key][()]
    for key in handle:
        obj = handle[key]
        if isinstance(obj, h5py.Dataset):
            return obj[()]
    raise KeyError("No dataset found in h5 file.")


def _tensor_from_pt(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        for key in FEATURE_KEYS:
            if key in obj:
                return _tensor_from_pt(obj[key])
        for value in obj.values():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return _tensor_from_pt(value)
    raise TypeError(f"Unsupported torch file content: {type(obj)!r}")


def load_array(path, preferred_h5_keys):
    suffix = Path(path).suffix.lower()
    if suffix in (".pt", ".pth"):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        return _tensor_from_pt(obj).float()
    if suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as handle:
            arr = _first_h5_dataset(handle, preferred_h5_keys)
        return torch.as_tensor(arr).float()
    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).float()
    if suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        return torch.from_numpy(data[key]).float()
    raise ValueError(f"Unsupported file type: {path}")


def _prepare_mri_volume(mri):
    mri = mri.float()
    if mri.ndim == 3:
        mri = mri.unsqueeze(0)
    elif mri.ndim == 4:
        if mri.shape[0] not in (1, 3) and mri.shape[-1] in (1, 3):
            mri = mri.permute(3, 0, 1, 2)
    else:
        raise ValueError(f"MRI volume must be 3D or 4D, got shape {tuple(mri.shape)}")
    mean = mri.mean()
    std = mri.std().clamp_min(1e-6)
    return (mri - mean) / std


class HER2Dataset(Dataset):
    def __init__(
        self,
        table,
        wsi_root,
        mri_root,
        use_mri_normalization=True,
    ):
        self.table = table.reset_index(drop=True)
        self.wsi_root = wsi_root
        self.mri_root = mri_root
        self.use_mri_normalization = use_mri_normalization

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        wsi_path = _resolve_file(self.wsi_root, row["wsi_id"])
        mri_path = _resolve_file(self.mri_root, row["mri_id"])

        wsi = load_array(wsi_path, FEATURE_KEYS)
        if wsi.ndim == 1:
            wsi = wsi.unsqueeze(0)
        if wsi.ndim != 2:
            wsi = wsi.view(wsi.shape[0], -1)

        mri = load_array(mri_path, VOLUME_KEYS)
        if self.use_mri_normalization:
            mri = _prepare_mri_volume(mri)

        label = int(row["label"])
        sample_id = f"{row['mri_id']}|{row['wsi_id']}"
        return {
            "wsi": wsi.float(),
            "mri": mri.float(),
            "label": label,
            "sample_id": sample_id,
            "wsi_path": wsi_path,
            "mri_path": mri_path,
        }


def collate_iada_batch(batch):
    wsi = [item["wsi"] for item in batch]
    mri = torch.stack([item["mri"] for item in batch], dim=0)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    sample_ids = [item["sample_id"] for item in batch]
    return {
        "wsi": wsi,
        "mri": mri,
        "label": labels,
        "sample_id": sample_ids,
    }
