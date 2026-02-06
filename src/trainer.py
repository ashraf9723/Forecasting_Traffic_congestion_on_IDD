
from src.model import TrafficGNN
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, train_data, adj_matrix, epochs=50, lr=0.001, val_data=None, val_adj=None):
    """
    model: The TrafficGNN model
    train_data: A tuple (X, Ext, Y)
    adj_matrix: The adjacency matrix A
    val_data: Optional tuple (X_val, Ext_val, Y_val)
    val_adj: Optional adjacency matrix for validation
    """
    X, Ext, Y = train_data
    dataset = TensorDataset(X, Ext, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    model.train()
    print("Starting Training...")

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_ext, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x, adj_matrix, batch_ext)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(loader)
        train_losses.append(avg_train_loss)

        # Validation loss
        if val_data is not None and val_adj is not None:
            model.eval()
            Xv, Extv, Yv = val_data
            with torch.no_grad():
                val_pred = model(Xv, val_adj, Extv)
                val_loss = criterion(val_pred, Yv).item()
            val_losses.append(val_loss)
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            model.train()
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        # Save best config for later use (after training loop)
        if best_config is not None:
            with open("best_config.json", "w") as f:
                json.dump(best_config, f)
            print(f"\nBest config: {best_config}")
            print(f"Best validation RMSE: {best_val_loss:.4f}")
            print(f"Best metrics: {best_metrics}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}")

    # Save last and best model
    torch.save(model.state_dict(), "traffic_gnn_model.pth")
    print("Model saved as traffic_gnn_model.pth")
    if best_model_state is not None:
        torch.save(best_model_state, "traffic_gnn_model_best.pth")
        print("Best model (by val loss) saved as traffic_gnn_model_best.pth")

    # Save losses for visualization
    import pickle
    with open('train_losses.pkl', 'wb') as f:
        pickle.dump(train_losses, f)
    if val_losses:
        with open('val_losses.pkl', 'wb') as f:
            pickle.dump(val_losses, f)



if __name__ == "__main__":
    # Configuration (should match your main.py)
    NODES = 50
    IN_DIM = 12
    EXT_DIM = 3
    HIDDEN = 64

    # --- Data loading and feature extraction ---
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model import TrafficGNN
    from src.graph_utils import get_adjacency_matrix
    import pandas as pd
    import glob
    import numpy as np
    import torch

    # Get project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Data paths
    obd_path = os.path.join(PROJECT_ROOT, 'data/idd_multimodal/supplement/obd/d0/obd.csv')
    lidar_dir = os.path.join(PROJECT_ROOT, 'data/idd_multimodal/supplement/lidar/d0/')

    # Load OBD speed data
    obd_df = pd.read_csv(obd_path)
    speed_seq = obd_df['speed'].values

    # Load and reshape LIDAR features to (NODES, IN_DIM)
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.npy')))
    if len(lidar_files) == 0:
        raise ValueError("No LIDAR .npy files found!")

    # Parameters for binning
    Z_TIERS = [-np.inf, -1, 0, 1, np.inf]  # 4 height bins

    # For temporal features
    prev_counts = None
    lidar_seq = []
    for idx, f in enumerate(lidar_files):
        arr = np.load(f)  # shape (N, 5)
        X_col, Y_col, Z_col, Int_col, Ring_col = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        # Bin by X into 50 equal-width bins
        x_min, x_max = X_col.min(), X_col.max()
        bins = np.linspace(x_min, x_max, NODES+1)
        node_features = np.zeros((NODES, IN_DIM), dtype=np.float32)
        for node in range(NODES):
            mask = (X_col >= bins[node]) & (X_col < bins[node+1])
            pts = arr[mask]
            if pts.shape[0] == 0:
                continue
            # 1-4: Point density in 4 Z tiers
            for zt in range(4):
                z_mask = (pts[:,2] >= Z_TIERS[zt]) & (pts[:,2] < Z_TIERS[zt+1])
                node_features[node, zt] = np.sum(z_mask)
            # 5: Mean intensity
            node_features[node, 4] = pts[:,3].mean()
            # 6: Std intensity
            node_features[node, 5] = pts[:,3].std() if pts.shape[0] > 1 else 0.0
            # 7: Variance X
            node_features[node, 6] = pts[:,0].var() if pts.shape[0] > 1 else 0.0
            # 8: Variance Y
            node_features[node, 7] = pts[:,1].var() if pts.shape[0] > 1 else 0.0
            # 9: Mean Z
            node_features[node, 8] = pts[:,2].mean()
            # 10: Std Z
            node_features[node, 9] = pts[:,2].std() if pts.shape[0] > 1 else 0.0
            # 11: Point count (for flow proxy)
            node_features[node, 10] = pts.shape[0]
        # 12: Point count diff vs previous frame
        if prev_counts is not None:
            node_features[:,11] = node_features[:,10] - prev_counts
        else:
            node_features[:,11] = 0.0
        prev_counts = node_features[:,10].copy()
        lidar_seq.append(node_features)
        if idx == 0:
            print(f"Sample node features for first frame (first 5 nodes):\n{node_features[:5]}")
    lidar_seq = np.stack(lidar_seq)  # (num_frames, 50, 12)
    print(f"LIDAR sequence shape for model: {lidar_seq.shape}")

    num_frames = lidar_seq.shape[0]
    num_nodes = lidar_seq.shape[1]

    # External features (dummy)
    ext_seq = np.zeros((num_frames, num_nodes, EXT_DIM), dtype=np.float32)

    # Target: truncate or pad speed_seq to num_frames
    if speed_seq.shape[0] >= num_frames:
        default_target = speed_seq[:num_frames]
    else:
        # Pad with last value if not enough
        pad = np.full((num_frames - speed_seq.shape[0],), speed_seq[-1])
        default_target = np.concatenate([speed_seq, pad])

    # Convert to tensors
    X = torch.tensor(lidar_seq, dtype=torch.float32)
    Ext = torch.tensor(ext_seq, dtype=torch.float32)
    Y = torch.tensor(default_target, dtype=torch.float32).reshape(num_frames, 1).repeat(1, num_nodes).unsqueeze(-1)

    # Train/val/test split (70/15/15)
    n = num_frames
    idx_train = int(n * 0.7)
    idx_val = int(n * 0.85)
    X_train, X_val, X_test = X[:idx_train], X[idx_train:idx_val], X[idx_val:]
    Ext_train, Ext_val, Ext_test = Ext[:idx_train], Ext[idx_train:idx_val], Ext[idx_val:]
    Y_train, Y_val, Y_test = Y[:idx_train], Y[idx_train:idx_val], Y[idx_val:]

    # Adjacency matrix


    coords = np.random.rand(num_nodes, 2)
    adj = get_adjacency_matrix(coords)

    # --- Evaluation function (must be defined before training loop) ---
    def evaluate(model, X, Ext, Y, adj):
        model.eval()
        with torch.no_grad():
            preds = model(X, adj, Ext)
            preds = preds.cpu().numpy().reshape(-1)
            targets = Y.cpu().numpy().reshape(-1)
            mae = np.mean(np.abs(preds - targets))
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return mae, rmse, r2

    # --- Hyperparameter grid search and training (always runs before visualization) ---
    search_space = {
        'lr': [0.01, 0.001],
        'hidden': [32, 64, 128]
    }
    best_val_loss = float('inf')
    best_config = None
    best_metrics = None
    best_model_state = None
    print("\nStarting hyperparameter grid search...")
    for lr in search_space['lr']:
        for hidden in search_space['hidden']:
            print(f"\n--- Training with lr={lr}, hidden={hidden} ---")
            model = TrafficGNN(IN_DIM, EXT_DIM, hidden)
            train_model(
                model,
                (X_train, Ext_train, Y_train),
                adj,
                epochs=5,
                lr=lr,
                val_data=(X_val, Ext_val, Y_val),
                val_adj=adj
            )
            # Evaluate best model for this config
            model.load_state_dict(torch.load("traffic_gnn_model_best.pth"))
            val_mae, val_rmse, val_r2 = evaluate(model, X_val, Ext_val, Y_val, adj)
            test_mae, test_rmse, test_r2 = evaluate(model, X_test, Ext_test, Y_test, adj)
            print(f"[Best Model] Validation: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}")
            print(f"[Best Model] Test:       MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
            # Track best config
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_config = {'lr': lr, 'hidden': hidden}
                best_metrics = {
                    'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
                    'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2
                }
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Save overall best model
    if best_model_state is not None:
        torch.save(best_model_state, "traffic_gnn_model_best_overall.pth")
        with open("best_config.json", "w") as f:
            json.dump(best_config, f)
        print(f"\nBest config: {best_config}")
        print(f"Best validation RMSE: {best_val_loss:.4f}")
        print(f"Best metrics: {best_metrics}")

    # --- Visualization: predictions vs ground truth ---
    import matplotlib.pyplot as plt
    print("\nVisualizing predictions vs ground truth (test set)...")
    try:
        with open("best_config.json", "r") as f:
            best_config = json.load(f)
        best_hidden = best_config["hidden"]
        best_model = TrafficGNN(IN_DIM, EXT_DIM, best_hidden)
        best_model.load_state_dict(torch.load("traffic_gnn_model_best_overall.pth"))
        best_model.eval()
        with torch.no_grad():
            preds = best_model(X_test, adj, Ext_test).cpu().numpy()  # (num_test, nodes, 1)
            targets = Y_test.cpu().numpy()  # (num_test, nodes, 1)
        # For visualization, average over nodes to get per-frame prediction
        preds_mean = preds.mean(axis=1).squeeze()
        targets_mean = targets.mean(axis=1).squeeze()
        # Plot a random sample of 100 frames
        import random
        idxs = sorted(random.sample(range(len(preds_mean)), min(100, len(preds_mean))))
        plt.figure(figsize=(12,6))
        plt.plot(idxs, targets_mean[idxs], label='Ground Truth', marker='o')
        plt.plot(idxs, preds_mean[idxs], label='Prediction', marker='x')
        plt.xlabel('Frame (sampled)')
        plt.ylabel('Speed (averaged over nodes)')
        plt.title('Predictions vs Ground Truth (Test Set)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print("best_config.json not found. Please run the training section first.")

    # --- Additional Visualizations ---
    # 1. Residuals (prediction error) vs. frame
    residuals = preds_mean - targets_mean
    plt.figure(figsize=(12,6))
    plt.plot(idxs, residuals[idxs], marker='o', linestyle='-', color='purple')
    plt.xlabel('Frame (sampled)')
    plt.ylabel('Prediction Error (Residual)')
    plt.title('Residuals (Prediction Error) vs Frame')
    plt.tight_layout()
    plt.show()

    # 2. Scatter plot: predictions vs. ground truth
    plt.figure(figsize=(8,8))
    plt.scatter(targets_mean[idxs], preds_mean[idxs], alpha=0.7)
    plt.plot([targets_mean[idxs].min(), targets_mean[idxs].max()], [targets_mean[idxs].min(), targets_mean[idxs].max()], 'r--')
    plt.xlabel('Ground Truth (sampled)')
    plt.ylabel('Prediction (sampled)')
    plt.title('Scatter: Prediction vs Ground Truth')
    plt.tight_layout()
    plt.show()

    # 3. Per-node predictions for a few selected nodes
    node_indices = [0, 1, 2, 3, 4]  # first 5 nodes
    plt.figure(figsize=(12,6))
    for node in node_indices:
        plt.plot(preds[:, node, 0], label=f'Pred Node {node}')
        plt.plot(targets[:, node, 0], '--', label=f'True Node {node}')
    plt.xlabel('Frame')
    plt.ylabel('Speed')
    plt.title('Per-Node Predictions vs Ground Truth (First 5 Nodes)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Histogram of prediction errors
    plt.figure(figsize=(8,6))
    plt.hist(residuals, bins=30, color='orange', alpha=0.7)
    plt.xlabel('Prediction Error (Residual)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.tight_layout()
    plt.show()

    # 5. Loss curves (train/val) over epochs
    # If available, plot from saved files or variables
    try:
        import pickle
        with open('train_losses.pkl', 'rb') as f:
            train_losses = pickle.load(f)
        with open('val_losses.pkl', 'rb') as f:
            val_losses = pickle.load(f)
        plt.figure(figsize=(10,6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print('Loss curves not available:', e)

    # Example usage (replace with your actual data/model)
    print("trainer.py loaded. This script only defines train_model().")

    # Runnable example
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.model import TrafficGNN
    from src.graph_utils import get_adjacency_matrix
    import pandas as pd
    import glob
    import numpy as np
    import torch

    # Get project root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Data paths
    obd_path = os.path.join(PROJECT_ROOT, 'data/idd_multimodal/supplement/obd/d0/obd.csv')
    lidar_dir = os.path.join(PROJECT_ROOT, 'data/idd_multimodal/supplement/lidar/d0/')

    # Configuration (should match your main.py)
    NODES = 50
    IN_DIM = 12
    EXT_DIM = 3
    HIDDEN = 64

    # Load OBD speed data
    obd_df = pd.read_csv(obd_path)
    speed_seq = obd_df['speed'].values

    # Load and reshape LIDAR features to (NODES, IN_DIM)
    lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.npy')))
    if len(lidar_files) == 0:
        raise ValueError("No LIDAR .npy files found!")

    # Parameters for binning
    NODES = 50
    IN_DIM = 12
    Z_TIERS = [-np.inf, -1, 0, 1, np.inf]  # 4 height bins

    # For temporal features
    prev_counts = None
    lidar_seq = []
    for idx, f in enumerate(lidar_files):
        arr = np.load(f)  # shape (N, 5)
        X_col, Y_col, Z_col, Int_col, Ring_col = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
        # Bin by X into 50 equal-width bins
        x_min, x_max = X_col.min(), X_col.max()
        bins = np.linspace(x_min, x_max, NODES+1)
        node_features = np.zeros((NODES, IN_DIM), dtype=np.float32)
        for node in range(NODES):
            mask = (X_col >= bins[node]) & (X_col < bins[node+1])
            pts = arr[mask]
            if pts.shape[0] == 0:
                continue
            # 1-4: Point density in 4 Z tiers
            for zt in range(4):
                z_mask = (pts[:,2] >= Z_TIERS[zt]) & (pts[:,2] < Z_TIERS[zt+1])
                node_features[node, zt] = np.sum(z_mask)
            # 5: Mean intensity
            node_features[node, 4] = pts[:,3].mean()
            # 6: Std intensity
            node_features[node, 5] = pts[:,3].std() if pts.shape[0] > 1 else 0.0
            # 7: Variance X
            node_features[node, 6] = pts[:,0].var() if pts.shape[0] > 1 else 0.0
            # 8: Variance Y
            node_features[node, 7] = pts[:,1].var() if pts.shape[0] > 1 else 0.0
            # 9: Mean Z
            node_features[node, 8] = pts[:,2].mean()
            # 10: Std Z
            node_features[node, 9] = pts[:,2].std() if pts.shape[0] > 1 else 0.0
            # 11: Point count (for flow proxy)
            node_features[node, 10] = pts.shape[0]
        # 12: Point count diff vs previous frame
        if prev_counts is not None:
            node_features[:,11] = node_features[:,10] - prev_counts
        else:
            node_features[:,11] = 0.0
        prev_counts = node_features[:,10].copy()
        lidar_seq.append(node_features)
        if idx == 0:
            print(f"Sample node features for first frame (first 5 nodes):\n{node_features[:5]}")
    lidar_seq = np.stack(lidar_seq)  # (num_frames, 50, 12)
    print(f"LIDAR sequence shape for model: {lidar_seq.shape}")

    num_frames = lidar_seq.shape[0]
    num_nodes = lidar_seq.shape[1]

    # External features (dummy)
    ext_seq = np.zeros((num_frames, num_nodes, EXT_DIM), dtype=np.float32)

    # Target: truncate or pad speed_seq to num_frames
    if speed_seq.shape[0] >= num_frames:
        default_target = speed_seq[:num_frames]
    else:
        # Pad with last value if not enough
        pad = np.full((num_frames - speed_seq.shape[0],), speed_seq[-1])
        default_target = np.concatenate([speed_seq, pad])

    # Convert to tensors
    X = torch.tensor(lidar_seq, dtype=torch.float32)
    Ext = torch.tensor(ext_seq, dtype=torch.float32)
    Y = torch.tensor(default_target, dtype=torch.float32).reshape(num_frames, 1).repeat(1, num_nodes).unsqueeze(-1)

    # Train/val/test split (70/15/15)
    n = num_frames
    idx_train = int(n * 0.7)
    idx_val = int(n * 0.85)
    X_train, X_val, X_test = X[:idx_train], X[idx_train:idx_val], X[idx_val:]
    Ext_train, Ext_val, Ext_test = Ext[:idx_train], Ext[idx_train:idx_val], Ext[idx_val:]
    Y_train, Y_val, Y_test = Y[:idx_train], Y[idx_train:idx_val], Y[idx_val:]

    # Adjacency matrix
    coords = np.random.rand(num_nodes, 2)
    adj = get_adjacency_matrix(coords)

    # --- Hyperparameter grid search ---
    def evaluate(model, X, Ext, Y, adj):
        model.eval()
        with torch.no_grad():
            preds = model(X, adj, Ext)
            preds = preds.cpu().numpy().reshape(-1)
            targets = Y.cpu().numpy().reshape(-1)
            mae = np.mean(np.abs(preds - targets))
            rmse = np.sqrt(np.mean((preds - targets) ** 2))
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return mae, rmse, r2

    search_space = {
        'lr': [0.01, 0.001],
        'hidden': [32, 64, 128]
    }
    best_val_loss = float('inf')
    best_config = None
    best_metrics = None
    best_model_state = None
    print("\nStarting hyperparameter grid search...")
    for lr in search_space['lr']:
        for hidden in search_space['hidden']:
            print(f"\n--- Training with lr={lr}, hidden={hidden} ---")
            model = TrafficGNN(IN_DIM, EXT_DIM, hidden)
            train_model(
                model,
                (X_train, Ext_train, Y_train),
                adj,
                epochs=5,
                lr=lr,
                val_data=(X_val, Ext_val, Y_val),
                val_adj=adj
            )
            # Evaluate best model for this config
            model.load_state_dict(torch.load("traffic_gnn_model_best.pth"))
            val_mae, val_rmse, val_r2 = evaluate(model, X_val, Ext_val, Y_val, adj)
            test_mae, test_rmse, test_r2 = evaluate(model, X_test, Ext_test, Y_test, adj)
            print(f"[Best Model] Validation: MAE={val_mae:.4f}, RMSE={val_rmse:.4f}, R2={val_r2:.4f}")
            print(f"[Best Model] Test:       MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, R2={test_r2:.4f}")
            # Track best config
            if val_rmse < best_val_loss:
                best_val_loss = val_rmse
                best_config = {'lr': lr, 'hidden': hidden}
                best_metrics = {
                    'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2,
                    'test_mae': test_mae, 'test_rmse': test_rmse, 'test_r2': test_r2
                }
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    # Save overall best model
    if best_model_state is not None:
        torch.save(best_model_state, "traffic_gnn_model_best_overall.pth")
        print(f"\nBest config: {best_config}")
        print(f"Best validation RMSE: {best_val_loss:.4f}")
        print(f"Best metrics: {best_metrics}")