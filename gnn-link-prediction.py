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

# ================================
# 4. Train the Model
# ================================

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the GCN model that produces 64-dimensional embeddings
model = GCNEncoder(dataset.num_features, 64).to(device)
# Move node features to the device
x = train_data.x.to(device)
# Move positive training edge index to the device
train_pos_edge_index = train_data.pos_edge_label_index.to(device)
# Set the optimizer (Adam) with learning rate 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train() # Set model to training mode
    optimizer.zero_grad() # Reset gradients before backpropagation
    
    # Use train_data, create node embedding
    z = model(train_data.x.to(device), train_data.edge_index.to(device))

    # Predict on positive (connected) edges
    pos_edge_index = train_data.pos_edge_label_index.to(device)
    pos_pred = decode(z, pos_edge_index)
    pos_label = torch.ones(pos_pred.size(0), device=device) # Create label 1 for all positive samples

    # Predict on negative (not connected) edges
    neg_edge_index = train_data.neg_edge_label_index.to(device)
    neg_pred = decode(z, neg_edge_index)
    neg_label = torch.zeros(neg_pred.size(0), device=device) # Create label 0 for all negative samples

    # Compute loss between predictions and labels
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    label = torch.cat([pos_label, neg_label], dim=0)
    loss = F.binary_cross_entropy_with_logits(pred, label)

    loss.backward() # Backpropagate to compute gradients
    optimizer.step() # Update model weights using gradients
    return loss # Return the training loss for monitoring

# ================================
# 5. Evaluate the Model
# ================================

@torch.no_grad()  # Don't track gradients during evaluation (for speed & memory)

def test():
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Again, disable gradient tracking
        # Compute node embeddings using trained model
        z = model(x, train_pos_edge_index) 
        
        # Predict scores for positive test edges (edges that actually exist)
        pos_pred = decode(z, test_data.pos_edge_label_index.to(device))
        
        # Predict scores for negative test edges (non-existing edges)
        neg_pred = decode(z, test_data.neg_edge_label_index.to(device))

        # Combine both positive and negative predictions into one tensor
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        # Create ground-truth labels: 1 for positive, 0 for negative
        label = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ]).to(device)

        # Classify predictions: if predicted score > 0, predict "link exists"
        acc = ((pred > 0).float() == label).sum() / label.size(0)

        return acc.item()
