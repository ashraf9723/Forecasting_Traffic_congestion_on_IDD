import glob
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.graph_utils import get_adjacency_matrix
from src.model import TrafficGNN


PROJECT_ROOT = Path(__file__).resolve().parent
NODES = 50
IN_DIM = 12
EXT_DIM = 3
IEEE_LABEL_SIZE = 10
IEEE_TICK_SIZE = 10
IEEE_LEGEND_SIZE = 10
IEEE_TITLE_SIZE = 10
PLOT_FIGSIZE = (15, 10)
PLOT_TITLE_SIZE = 24
PLOT_AXIS_LABEL_SIZE = 20
PLOT_TICK_LABEL_SIZE = 16
PLOT_LEGEND_SIZE = 18
PLOT_LINE_WIDTH = 3
PLOT_MARKER_SIZE = 10


def style_current_axes() -> None:
    plt.xticks(rotation=20, fontsize=PLOT_TICK_LABEL_SIZE)
    plt.yticks(fontsize=PLOT_TICK_LABEL_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=PLOT_TICK_LABEL_SIZE)
    plt.grid(True)


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": IEEE_LABEL_SIZE,
            "axes.labelsize": IEEE_LABEL_SIZE,
            "axes.titlesize": IEEE_TITLE_SIZE,
            "xtick.labelsize": IEEE_TICK_SIZE,
            "ytick.labelsize": IEEE_TICK_SIZE,
            "legend.fontsize": IEEE_LEGEND_SIZE,
            "figure.titlesize": IEEE_TITLE_SIZE,
            "savefig.dpi": 300,
        }
    )


def load_features_and_targets():
    obd_path = PROJECT_ROOT / "data" / "idd_multimodal" / "supplement" / "obd" / "d0" / "obd.csv"
    lidar_dir = PROJECT_ROOT / "data" / "idd_multimodal" / "supplement" / "lidar" / "d0"

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
        bins = np.linspace(x_col.min(), x_col.max(), NODES + 1)
        node_features = np.zeros((NODES, IN_DIM), dtype=np.float32)

        for node in range(NODES):
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

    ext_seq = np.zeros((num_frames, num_nodes, EXT_DIM), dtype=np.float32)

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
    x_test = x_tensor[idx_val_start:]
    ext_test = ext_tensor[idx_val_start:]
    y_test = y_tensor[idx_val_start:]

    return x_test, ext_test, y_test


def load_model_and_predictions(x_test, ext_test, y_test):
    best_config_path = PROJECT_ROOT / "best_config.json"
    if best_config_path.exists():
        with open(best_config_path, "r", encoding="utf-8") as file:
            best_config = json.load(file)
        hidden_dim = int(best_config.get("hidden", 64))
    else:
        hidden_dim = 64

    model = TrafficGNN(IN_DIM, EXT_DIM, hidden_dim)
    state_path = PROJECT_ROOT / "traffic_gnn_model_best_overall.pth"
    if not state_path.exists():
        state_path = PROJECT_ROOT / "traffic_gnn_model_best.pth"
    if not state_path.exists():
        state_path = PROJECT_ROOT / "traffic_gnn_model.pth"

    if not state_path.exists():
        raise RuntimeError("No model checkpoint found in project root.")

    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    np.random.seed(42)
    coords = np.random.rand(y_test.shape[1], 2)
    adj = get_adjacency_matrix(coords)

    with torch.no_grad():
        preds = model(x_test, adj, ext_test).cpu().numpy()
        targets = y_test.cpu().numpy()

    preds_mean = preds.mean(axis=1).squeeze()
    targets_mean = targets.mean(axis=1).squeeze()

    random.seed(42)
    sampled_count = min(100, len(preds_mean))
    idxs = sorted(random.sample(range(len(preds_mean)), sampled_count))
    residuals = preds_mean - targets_mean

    return preds, targets, preds_mean, targets_mean, residuals, idxs


def save_figure_1(targets_mean, preds_mean, idxs):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(
        idxs,
        targets_mean[idxs],
        label="Ground Truth",
        marker="o",
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
    )
    plt.plot(
        idxs,
        preds_mean[idxs],
        label="Prediction",
        marker="x",
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
    )
    plt.xlabel("Frame (sampled)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Speed (avg over nodes)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Predictions vs Ground Truth", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_1.png", bbox_inches="tight")
    plt.close(fig)


def save_figure_2(residuals, idxs):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(
        idxs,
        residuals[idxs],
        marker="o",
        linestyle="-",
        color="purple",
        linewidth=PLOT_LINE_WIDTH,
        markersize=PLOT_MARKER_SIZE,
    )
    plt.xlabel("Frame (sampled)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Prediction Error (Residual)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Residuals vs Frame", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_2.png", bbox_inches="tight")
    plt.close(fig)


def save_figure_3(targets_mean, preds_mean, idxs):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    plt.scatter(targets_mean[idxs], preds_mean[idxs], alpha=0.7, s=100)
    line_min = targets_mean[idxs].min()
    line_max = targets_mean[idxs].max()
    plt.plot([line_min, line_max], [line_min, line_max], "r--", linewidth=PLOT_LINE_WIDTH)
    plt.xlabel("Ground Truth (sampled)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Prediction (sampled)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Scatter: Prediction vs Ground Truth", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_3 Scatter.png", bbox_inches="tight")
    plt.close(fig)


def save_figure_4(preds, targets):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    for node in [0, 1, 2, 3, 4]:
        plt.plot(preds[:, node, 0], label=f"Pred Node {node}", linewidth=PLOT_LINE_WIDTH)
        plt.plot(targets[:, node, 0], "--", label=f"True Node {node}", linewidth=PLOT_LINE_WIDTH)
    plt.xlabel("Frame", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Speed", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Per-Node Predictions vs Ground Truth", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.legend(ncol=2, frameon=False, fontsize=PLOT_LEGEND_SIZE)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_4 Node prediction.png", bbox_inches="tight")
    plt.close(fig)


def save_figure_5(residuals):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    plt.hist(residuals, bins=30, color="orange", alpha=0.7)
    plt.xlabel("Prediction Error (Residual)", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Frequency", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Histogram of Prediction Errors", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_5 Histogram_of_prediction error.png", bbox_inches="tight")
    plt.close(fig)


def save_pernode_figure(preds, targets):
    fig = plt.figure(figsize=PLOT_FIGSIZE)
    for node in [0, 10, 20, 30, 40]:
        plt.plot(preds[:, node, 0], label=f"Pred N{node}", linewidth=PLOT_LINE_WIDTH)
        plt.plot(targets[:, node, 0], "--", label=f"True N{node}", linewidth=PLOT_LINE_WIDTH)
    plt.xlabel("Frame", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Speed", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Per-node Comparison", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.legend(ncol=2, frameon=False, fontsize=PLOT_LEGEND_SIZE)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "Figure_Pernode.png", bbox_inches="tight")
    plt.close(fig)


def save_training_validation_figure():
    train_path = PROJECT_ROOT / "train_losses.pkl"
    val_path = PROJECT_ROOT / "val_losses.pkl"
    if not train_path.exists() or not val_path.exists():
        return

    import pickle

    with open(train_path, "rb") as file:
        train_losses = pickle.load(file)
    with open(val_path, "rb") as file:
        val_losses = pickle.load(file)

    fig = plt.figure(figsize=PLOT_FIGSIZE)
    plt.plot(train_losses, label="Train Loss", linewidth=PLOT_LINE_WIDTH)
    plt.plot(val_losses, label="Val Loss", linewidth=PLOT_LINE_WIDTH)
    plt.xlabel("Epoch", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.ylabel("Loss", fontsize=PLOT_AXIS_LABEL_SIZE)
    plt.title("Training and Validation Loss Curves", fontsize=PLOT_TITLE_SIZE)
    style_current_axes()
    plt.legend(frameon=False, fontsize=PLOT_LEGEND_SIZE)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "trainig and Validation.png", bbox_inches="tight")
    plt.close(fig)


def main():
    set_ieee_style()
    x_test, ext_test, y_test = load_features_and_targets()
    preds, targets, preds_mean, targets_mean, residuals, idxs = load_model_and_predictions(
        x_test, ext_test, y_test
    )

    save_figure_1(targets_mean, preds_mean, idxs)
    save_figure_2(residuals, idxs)
    save_figure_3(targets_mean, preds_mean, idxs)
    save_figure_4(preds, targets)
    save_figure_5(residuals)
    save_pernode_figure(preds, targets)
    save_training_validation_figure()

    print("Regenerated IEEE-style figures:")
    print("- Figure_1.png")
    print("- Figure_2.png")
    print("- Figure_3 Scatter.png")
    print("- Figure_4 Node prediction.png")
    print("- Figure_5 Histogram_of_prediction error.png")
    print("- Figure_Pernode.png")
    print("- trainig and Validation.png")


if __name__ == "__main__":
    main()