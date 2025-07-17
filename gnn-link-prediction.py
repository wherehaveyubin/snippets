# ================================
# 0. Import Required Libraries
# ================================

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit

# ================================
# 1. Load Dataset
# ================================

dataset = Planetoid(root='data', name='Cora')
data = dataset[0] 

# --------------------------------------
# Load the Cora dataset.
# Cora is a graph consisting of papers and citation relationships.
# - Nodes: Papers
# - Edges: Citation relationships between papers
# --------------------------------------

# Split the edges into train/validation/test sets for link prediction
transform = RandomLinkSplit(
    is_undirected=True,
    split_labels=True, # Create pos_edge_label_index
    add_negative_train_samples=True 
)

train_data, val_data, test_data = transform(data)

print("=== Dataset Summary ===")
print(dataset)

print("\n=== Raw Graph Data ===")
print(data)

print("\n=== Applied Transform ===")
print(transform)

print("\n=== Train Data ===")
print(train_data)

print("\n=== Validation Data ===")
print(val_data)

print("\n=== Test Data ===")
print(test_data)

"""
=== Dataset Summary ===
Cora()

=== Raw Graph Data ===
Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

=== Applied Transform ===
RandomLinkSplit(num_val=0.1, num_test=0.2)

=== Train Data ===
Data(x=[2708, 1433], edge_index=[2, 7392], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[7392], edge_label_index=[2, 7392])

=== Validation Data ===
Data(x=[2708, 1433], edge_index=[2, 7392], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[1054], edge_label_index=[2, 1054])

=== Test Data ===
Data(x=[2708, 1433], edge_index=[2, 8446], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708], edge_label=[2110], edge_label_index=[2, 2110])
"""

# ================================
# 2. Define GCN (Graph Convolutional Network) Model
# ================================

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        
        # First GCN layer: expands hidden dimension
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        
        # Second GCN layer: projects to output embedding size
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # Apply first GCN layer and ReLU activation
        x = self.conv1(x, edge_index)
        x = F.relu(x) 
        
        # Apply second GCN layer
        x = self.conv2(x, edge_index)
        
        return x # Return final node embeddings

# ================================
# 3. Link Prediction Decoder
# ================================

def decode(z, edge_index):
    return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

