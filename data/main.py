import sys
import os
import pandas as pd
import glob
import numpy as np
import torch
import torch.utils.data as data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import TrafficGNN
from src.graph_utils import get_adjacency_matrix

# Get script and data directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
NODES = 50 # Example number of road segments
IN_DIM = 12 # 12 past time steps
EXT_DIM = 3  # Weather, AQI, Event
HIDDEN = 64

# Initialize
model = TrafficGNN(IN_DIM, EXT_DIM, HIDDEN)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load OBD speed data
obd_path = os.path.join(SCRIPT_DIR, 'idd_multimodal/supplement/obd/d0/obd.csv')
obd_df = pd.read_csv(obd_path)
speed_seq = obd_df['speed'].values  # (num_frames,)

# Load LIDAR features (assuming each .npy is a frame, shape: (NODES, IN_DIM))
lidar_dir = os.path.join(SCRIPT_DIR, 'idd_multimodal/supplement/lidar/d0/')
lidar_files = sorted(glob.glob(os.path.join(lidar_dir, '*.npy')))
lidar_seq = [np.load(f) for f in lidar_files]

# Check shapes and handle variable sizes
print(f"Loaded {len(lidar_seq)} LIDAR frames")
if lidar_seq:
    shapes = [arr.shape for arr in lidar_seq[:5]]
    print(f"First 5 shapes: {shapes}")
    # Find max dimensions and pad if needed
    max_shape = tuple(max(arr.shape[i] if i < len(arr.shape) else 0 for arr in lidar_seq) 
                      for i in range(max(len(arr.shape) for arr in lidar_seq)))
    print(f"Max shape: {max_shape}")
    
    # Pad arrays to same shape
    padded_seq = []
    for arr in lidar_seq:
        if arr.shape != max_shape:
            pad_width = [(0, max_shape[i] - arr.shape[i]) if i < len(arr.shape) else (0, 0) 
                        for i in range(len(max_shape))]
            arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
        padded_seq.append(arr)
    lidar_seq = np.stack(padded_seq)  # (num_frames, NODES, IN_DIM)
    print(f"Final stacked shape: {lidar_seq.shape}")

# Parameters
BATCH_SIZE = 32
EPOCHS = 5

# Dummy external features (all zeros)
ext_seq = np.zeros((len(lidar_seq), lidar_seq.shape[1], EXT_DIM))

# Dataset class
default_target = speed_seq[:len(lidar_seq)]  # Align lengths if needed
class TrafficDataset(data.Dataset):
    def __init__(self, lidar_seq, ext_seq, target_seq):
        self.lidar_seq = lidar_seq
        self.ext_seq = ext_seq
        self.target_seq = target_seq
    def __len__(self):
        return len(self.lidar_seq)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.lidar_seq[idx], dtype=torch.float32),
            torch.tensor(self.ext_seq[idx], dtype=torch.float32),
            torch.tensor(self.target_seq[idx], dtype=torch.float32)
        )

dataset = TrafficDataset(lidar_seq, ext_seq, default_target)
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Use random coordinates for adjacency
coords = np.random.rand(lidar_seq.shape[1], 2)
adj = get_adjacency_matrix(coords)

# Training loop
loss_fn = torch.nn.MSELoss()
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_ext, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x, adj, batch_ext)  # (batch, nodes, hidden)
        # For demo, predict speed at node 0 (or mean over nodes)
        pred = output[:, 0, 0] if output.shape[2] > 0 else output.mean(dim=1).squeeze()
        loss = loss_fn(pred, batch_y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")