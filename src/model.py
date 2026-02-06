import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeAttentionLayer(nn.Module):
    def __init__(self, node_feat_dim, ext_dim, hidden_dim):
        super(KnowledgeAttentionLayer, self).__init__()
        self.fc_fusion = nn.Linear(node_feat_dim + ext_dim, hidden_dim)
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x, external_info):
        # x: (Batch, Nodes, Feats) -> Historical Traffic
        # external_info: (Batch, Nodes, Ext_Feats) -> Weather/AQI
        combined = torch.cat([x, external_info], dim=-1)
        fused = F.relu(self.fc_fusion(combined))
        
        # Calculate importance of external info
        score = torch.sigmoid(self.attn_weights(fused))
        return fused * score

class TrafficGNN(nn.Module):
    def __init__(self, in_dim, ext_dim, hidden_dim):
        super(TrafficGNN, self).__init__()
        self.attention = KnowledgeAttentionLayer(in_dim, ext_dim, hidden_dim)
        # Simple Graph Convolutional layer logic
        self.gcn = nn.Linear(hidden_dim, hidden_dim)
        # Output layer to map hidden_dim to 1 per node
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj, external_info):
        # 1. Knowledge Fusion
        h = self.attention(x, external_info)
        # 2. Spatial Message Passing (h_new = A * h * W)
        h = torch.matmul(adj, h)
        h = F.relu(self.gcn(h))
        # 3. Output prediction per node
        out = self.out(h)
        return out