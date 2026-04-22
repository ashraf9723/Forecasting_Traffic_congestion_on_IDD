import glob
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.graph_utils import get_adjacency_matrix
from src.model import TrafficGNN


def load_test_split(project_root: Path, nodes: int = 50, in_dim: int = 12, ext_dim: int = 3):
    obd_path = project_root / "data" / "idd_multimodal" / "supplement" / "obd" / "d0" / "obd.csv"
    lidar_dir = project_root / "data" / "idd_multimodal" / "supplement" / "lidar" / "d0"

    obd_df = pd.read_csv(obd_path)
    speed_seq = obd_df["speed"].values

    lidar_files = sorted(
        [
            f
            for f in glob.glob(str(lidar_dir / "*.npy"))
            if not os.path.basename(f).startswith("._")
        ]
    )
    if not lidar_files:
        raise RuntimeError("No LIDAR .npy files found in d0 directory.")

    z_tiers = [-np.inf, -1, 0, 1, np.inf]
    prev_counts = None
    lidar_seq = []

    for file_path in lidar_files:
        arr = np.load(file_path)
        x_col = arr[:, 0]
        bins = np.linspace(x_col.min(), x_col.max(), nodes + 1)
        node_features = np.zeros((nodes, in_dim), dtype=np.float32)

        for node in range(nodes):
            mask = (x_col >= bins[node]) & (x_col < bins[node + 1])
            pts = arr[mask]
            if pts.shape[0] == 0:
                continue

            for zt in range(4):
                z_mask = (pts[:, 2] >= z_tiers[zt]) & (pts[:, 2] < z_tiers[zt + 1])
                node_features[node, zt] = np.sum(z_mask)

            node_features[node, 4] = pts[:, 3].mean()
            node_features[node, 5] = pts[:, 3].std() if pts.shape[0] > 1 else 0.0
            node_features[node, 6] = pts[:, 0].var() if pts.shape[0] > 1 else 0.0
            node_features[node, 7] = pts[:, 1].var() if pts.shape[0] > 1 else 0.0
            node_features[node, 8] = pts[:, 2].mean()
            node_features[node, 9] = pts[:, 2].std() if pts.shape[0] > 1 else 0.0
            node_features[node, 10] = pts.shape[0]

        if prev_counts is not None:
            node_features[:, 11] = node_features[:, 10] - prev_counts
        else:
            node_features[:, 11] = 0.0
        prev_counts = node_features[:, 10].copy()
        lidar_seq.append(node_features)

    lidar_seq = np.stack(lidar_seq)
    num_frames, num_nodes, _ = lidar_seq.shape

    ext_seq = np.zeros((num_frames, num_nodes, ext_dim), dtype=np.float32)

    if speed_seq.shape[0] >= num_frames:
        target = speed_seq[:num_frames]
    else:
        pad = np.full((num_frames - speed_seq.shape[0],), speed_seq[-1])
        target = np.concatenate([speed_seq, pad])

    x_tensor = torch.tensor(lidar_seq, dtype=torch.float32)
    ext_tensor = torch.tensor(ext_seq, dtype=torch.float32)
    y_tensor = (
        torch.tensor(target, dtype=torch.float32)
        .reshape(num_frames, 1)
        .repeat(1, num_nodes)
        .unsqueeze(-1)
    )

    idx_val_start = int(num_frames * 0.85)

    return {
        "x_test": x_tensor[idx_val_start:],
        "ext_test": ext_tensor[idx_val_start:],
        "y_test": y_tensor[idx_val_start:],
        "num_frames": int(num_frames),
        "num_nodes": int(num_nodes),
        "test_start_idx": int(idx_val_start),
    }


def compute_metrics(preds: np.ndarray, targets: np.ndarray):
    pred_flat = preds.reshape(-1)
    true_flat = targets.reshape(-1)
    err = pred_flat - true_flat

    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / np.clip(np.abs(true_flat), 1e-8, None)) * 100.0)

    ss_res = float(np.sum((true_flat - pred_flat) ** 2))
    ss_tot = float(np.sum((true_flat - np.mean(true_flat)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    acc_05 = float(np.mean(np.abs(err) <= 0.5) * 100.0)
    acc_10 = float(np.mean(np.abs(err) <= 1.0) * 100.0)

    preds_mean = preds.mean(axis=1).squeeze()
    true_mean = targets.mean(axis=1).squeeze()
    frame_err = preds_mean - true_mean

    frame_mse = float(np.mean(frame_err**2))
    frame_rmse = float(np.sqrt(frame_mse))
    frame_mae = float(np.mean(np.abs(frame_err)))

    frame_ss_res = float(np.sum((true_mean - preds_mean) ** 2))
    frame_ss_tot = float(np.sum((true_mean - np.mean(true_mean)) ** 2))
    frame_r2 = float(1 - frame_ss_res / frame_ss_tot) if frame_ss_tot > 0 else float("nan")

    def to_class(x):
        return np.where(x <= 3, 0, np.where(x <= 6, 1, 2))

    y_true_cls = to_class(true_flat)
    y_pred_cls = to_class(pred_flat)

    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true_cls, y_pred_cls):
        cm[t, p] += 1

    labels = ["Low (<=3)", "Moderate (3-6]", "High (>6)"]
    per_class = {}
    for c, name in enumerate(labels):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())

        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)

        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    overall_acc_cls = float(np.mean(y_true_cls == y_pred_cls) * 100.0)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "regression_metrics_all_points": {
            "mae": mae,
            "rmse": rmse,
            "mse": mse,
            "mape_percent": mape,
            "r2": r2,
            "accuracy_abs_err_le_0_5_percent": acc_05,
            "accuracy_abs_err_le_1_0_percent": acc_10,
        },
        "regression_metrics_frame_mean": {
            "mae": frame_mae,
            "rmse": frame_rmse,
            "mse": frame_mse,
            "r2": frame_r2,
        },
        "classification_3bin": {
            "bins": labels,
            "overall_accuracy_percent": overall_acc_cls,
            "macro_f1": macro_f1,
            "confusion_matrix_rows_true_cols_pred": cm.tolist(),
            "per_class": per_class,
        },
    }


def main():
    project_root = Path(__file__).resolve().parent
    nodes, in_dim, ext_dim = 50, 12, 3

    split = load_test_split(project_root, nodes=nodes, in_dim=in_dim, ext_dim=ext_dim)

    best_config_path = project_root / "best_config.json"
    hidden_dim = 64
    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as file:
            best_config = json.load(file)
        hidden_dim = int(best_config.get("hidden", 64))

    model = TrafficGNN(in_dim, ext_dim, hidden_dim)

    state_path = project_root / "traffic_gnn_model_best_overall.pth"
    if not state_path.exists():
        state_path = project_root / "traffic_gnn_model_best.pth"
    if not state_path.exists():
        state_path = project_root / "traffic_gnn_model.pth"
    if not state_path.exists():
        raise RuntimeError("No model checkpoint found in project root.")

    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    np.random.seed(42)
    coords = np.random.rand(split["y_test"].shape[1], 2)
    adj = get_adjacency_matrix(coords)

    with torch.no_grad():
        preds = model(split["x_test"], adj, split["ext_test"]).cpu().numpy()
        targets = split["y_test"].cpu().numpy()

    metrics = compute_metrics(preds, targets)

    output = {
        "date": str(date.today()),
        "data": {
            "total_frames": split["num_frames"],
            "test_frames": int(split["x_test"].shape[0]),
            "nodes": split["num_nodes"],
            "test_points": int(targets.reshape(-1).shape[0]),
            "split": {"train_val_test": "70/15/15", "test_start_index": split["test_start_idx"]},
        },
        "model": {
            "checkpoint": str(state_path.name),
            "hidden_dim": int(hidden_dim),
            "in_dim": in_dim,
            "ext_dim": ext_dim,
        },
        **metrics,
    }

    out_path = project_root / "results_evaluated_2026-02-18.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(json.dumps(output, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
