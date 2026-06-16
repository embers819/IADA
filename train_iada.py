import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import HER2Dataset, collate_iada_batch, read_label_csv
from models import IADA
from models.iada import apply_dgm, dgm_coefficients
from utils import FocalLoss, binary_metrics, multiclass_metrics, seed_everything


def _column_arg(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate IADA for HER2 prediction.")
    parser.add_argument("--config", default=None, help="Optional YAML config file.")
    parser.add_argument("--mode", default="internal_5fold", choices=["internal_5fold", "external"])

    parser.add_argument("--internal_csv", default=None)
    parser.add_argument("--external_csv", default=None)
    parser.add_argument("--wsi_root", default=None)
    parser.add_argument("--mri_root", default=None)
    parser.add_argument("--output_dir", default="runs/iada")
    parser.add_argument("--csv_has_header", action="store_true")
    parser.add_argument("--mri_col", default="0")
    parser.add_argument("--wsi_col", default="1")
    parser.add_argument("--label_col", default="2")

    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--wsi_input_dim", type=int, default=1024)
    parser.add_argument("--wsi_embed_dim", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--rrt_region_num", type=int, default=8)
    parser.add_argument("--rrt_layers", type=int, default=2)
    parser.add_argument("--rrt_heads", type=int, default=8)

    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lambda_reg", type=float, default=2e-2)
    parser.add_argument("--focal_gamma", type=float, default=1.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--use_dgm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    return parser


def parse_args():
    parser = build_parser()
    prelim, _ = parser.parse_known_args()
    if prelim.config:
        import yaml

        with open(prelim.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        parser.set_defaults(**config)
    args = parser.parse_args()
    args.mri_col = _column_arg(args.mri_col)
    args.wsi_col = _column_arg(args.wsi_col)
    args.label_col = _column_arg(args.label_col)
    return args


def make_model(args, device):
    model = IADA(
        n_classes=args.n_classes,
        wsi_input_dim=args.wsi_input_dim,
        wsi_embed_dim=args.wsi_embed_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        rrt_region_num=args.rrt_region_num,
        rrt_layers=args.rrt_layers,
        rrt_heads=args.rrt_heads,
    )
    return model.to(device)


def make_loader(table, args, shuffle):
    dataset = HER2Dataset(table, args.wsi_root, args.mri_root)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_iada_batch,
    )


def move_batch(batch, device):
    return {
        "wsi": [item.to(device, non_blocking=True) for item in batch["wsi"]],
        "mri": batch["mri"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
        "sample_id": batch["sample_id"],
    }


def train_one_epoch(model, loader, optimizer, criterion, args, device, epoch):
    model.train()
    running = {"loss": [], "cls_loss": [], "dist_loss": [], "align_loss": [], "reg_loss": []}
    dgm_stats = []

    iterator = tqdm(loader, desc=f"epoch {epoch:03d}", leave=False)
    for batch in iterator:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(batch["wsi"], batch["mri"])
        losses = model.compute_loss(
            outputs,
            batch["label"],
            criterion,
            alpha=args.alpha,
            lambda_reg=args.lambda_reg,
        )
        losses["loss"].backward()

        if args.use_dgm:
            coeffs = dgm_coefficients(
                outputs["wsi_logits"],
                outputs["mri_logits"],
                batch["label"],
                beta=args.beta,
            )
            apply_dgm(model, coeffs)
            dgm_stats.append(coeffs)

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        for key in running:
            value = losses[key]
            if torch.is_tensor(value):
                value = value.item()
            running[key].append(float(value))
        iterator.set_postfix(loss=np.mean(running["loss"]))

    summary = {key: float(np.mean(values)) for key, values in running.items()}
    if dgm_stats:
        for key in ("ratio_wsi", "ratio_mri", "coeff_wsi", "coeff_mri"):
            summary[key] = float(np.mean([item[key] for item in dgm_stats]))
    return summary


@torch.no_grad()
def evaluate(model, loader, criterion, args, device):
    model.eval()
    y_true = []
    prob_rows = []
    sample_ids = []
    losses = []

    for batch in tqdm(loader, desc="evaluate", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch["wsi"], batch["mri"])
        loss = model.compute_loss(
            outputs,
            batch["label"],
            criterion,
            alpha=args.alpha,
            lambda_reg=args.lambda_reg,
        )["loss"]
        prob = torch.softmax(outputs["logits"], dim=1)
        y_true.extend(batch["label"].detach().cpu().numpy().astype(int).tolist())
        prob_rows.extend(prob.detach().cpu().numpy().tolist())
        sample_ids.extend(batch["sample_id"])
        losses.append(float(loss.item()))

    prob_arr = np.asarray(prob_rows)
    if args.n_classes == 2:
        metrics = binary_metrics(y_true, prob_arr[:, 1])
    else:
        metrics = multiclass_metrics(y_true, prob_arr)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    pred = pd.DataFrame({"sample_id": sample_ids, "label": y_true})
    for cls_idx in range(prob_arr.shape[1]):
        pred[f"prob_{cls_idx}"] = prob_arr[:, cls_idx]
    pred["pred"] = prob_arr.argmax(axis=1)
    return metrics, pred


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def fit_split(train_table, eval_table, args, run_dir, split_name, use_eval_for_selection=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(train_table, args, shuffle=True)
    eval_loader = make_loader(eval_table, args, shuffle=False) if eval_table is not None else None
    model = make_model(args, device)
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -np.inf
    best_epoch = None
    history = []
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, args, device, epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_log.items()}}

        if use_eval_for_selection and eval_loader is not None:
            eval_metrics, eval_pred = evaluate(model, eval_loader, criterion, args, device)
            row.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            auc = eval_metrics.get("auc", float("nan"))
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, best_path)
                eval_pred.to_csv(run_dir / "best_predictions.csv", index=False)
                write_json(run_dir / "best_metrics.json", eval_metrics)

        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    torch.save({"model": model.state_dict(), "args": vars(args), "epoch": args.epochs}, last_path)

    if eval_loader is None:
        final_metrics = {"best_epoch": args.epochs}
        checkpoint_path = last_path
    elif use_eval_for_selection and best_path.exists():
        checkpoint_path = best_path
        final_metrics = json.load(open(run_dir / "best_metrics.json", "r", encoding="utf-8"))
        final_metrics["best_epoch"] = best_epoch
    else:
        checkpoint_path = last_path
        final_metrics, final_pred = evaluate(model, eval_loader, criterion, args, device)
        final_pred.to_csv(run_dir / "last_predictions.csv", index=False)
        write_json(run_dir / "last_metrics.json", final_metrics)
        final_metrics["best_epoch"] = args.epochs

    final_metrics["split"] = split_name
    final_metrics["checkpoint"] = str(checkpoint_path)
    write_json(run_dir / "summary.json", final_metrics)
    return final_metrics, checkpoint_path


@torch.no_grad()
def test_checkpoint(checkpoint_path, test_table, args, run_dir, split_name):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = make_model(args, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    loader = make_loader(test_table, args, shuffle=False)
    metrics, pred = evaluate(model, loader, criterion, args, device)
    metrics["split"] = split_name
    pred.to_csv(Path(run_dir) / f"{split_name}_predictions.csv", index=False)
    write_json(Path(run_dir) / f"{split_name}_metrics.json", metrics)
    return metrics


def run_internal_5fold(table, args):
    if args.folds != 5:
        raise ValueError("HER2 internal evaluation is fixed to 5-fold unless explicitly changing project protocol.")
    labels = table["label"].to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        split_dir = Path(args.output_dir) / f"fold_{fold}"
        train_table = table.iloc[train_idx].reset_index(drop=True)
        val_table = table.iloc[val_idx].reset_index(drop=True)
        metrics, _ = fit_split(
            train_table,
            val_table,
            args,
            split_dir,
            split_name=f"fold_{fold}",
            use_eval_for_selection=True,
        )
        metrics["fold"] = fold
        fold_metrics.append(metrics)

    df = pd.DataFrame(fold_metrics)
    df.to_csv(Path(args.output_dir) / "internal_5fold_metrics.csv", index=False)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != "fold"]
    summary = {}
    for col in numeric_cols:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    write_json(Path(args.output_dir) / "internal_5fold_summary.json", summary)
    return summary


def run_external(internal_table, external_table, args):
    split_dir = Path(args.output_dir) / "external"
    _, checkpoint = fit_split(
        internal_table,
        None,
        args,
        split_dir,
        split_name="external_train_all_internal",
        use_eval_for_selection=False,
    )
    return test_checkpoint(checkpoint, external_table, args, split_dir, "external_test")


def main():
    args = parse_args()
    seed_everything(args.seed)
    required = ["internal_csv", "wsi_root", "mri_root"]
    if args.mode == "external":
        required.append("external_csv")
    missing = [name for name in required if getattr(args, name) in (None, "")]
    if missing:
        raise ValueError(f"Missing required argument(s): {', '.join(missing)}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.output_dir) / "config.json", vars(args))

    internal_table = read_label_csv(
        args.internal_csv,
        has_header=args.csv_has_header,
        mri_col=args.mri_col,
        wsi_col=args.wsi_col,
        label_col=args.label_col,
    )

    if args.mode == "internal_5fold":
        summary = run_internal_5fold(internal_table, args)
    else:
        external_table = read_label_csv(
            args.external_csv,
            has_header=args.csv_has_header,
            mri_col=args.mri_col,
            wsi_col=args.wsi_col,
            label_col=args.label_col,
        )
        summary = run_external(internal_table, external_table, args)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
