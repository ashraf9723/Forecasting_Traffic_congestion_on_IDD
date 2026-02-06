import numpy as np
import torch

def compute_dist_matrix(coords):
    # coords: (N, 2) array of [lat, long]
    dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis] - coords[np.newaxis, :])**2, axis=2))
    return dist_matrix

def get_adjacency_matrix(coords, sigma=0.1, threshold=0.5):
    dist_matrix = compute_dist_matrix(coords)
    # Gaussian kernel for weights
    adj = np.exp(-dist_matrix**2 / sigma**2)
    adj[adj < threshold] = 0
    return torch.tensor(adj, dtype=torch.float32)
