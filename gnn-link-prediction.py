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

# ================================
# 6. Run Training
# ================================

for epoch in range(1, 201): # Loop over 200 training epochs
    loss = train()
    if epoch % 20 == 0: # Every 20 epochs, evaluate the model
        acc = test() # Evaluate link prediction accuracy
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}')
       
"""
Epoch 020, Loss: 0.4149, Test Acc: 0.7265
Epoch 040, Loss: 0.2013, Test Acc: 0.6867
Epoch 060, Loss: 0.0424, Test Acc: 0.6749
Epoch 080, Loss: 0.0044, Test Acc: 0.6730
Epoch 100, Loss: 0.0009, Test Acc: 0.6697
Epoch 120, Loss: 0.0004, Test Acc: 0.6716
Epoch 140, Loss: 0.0003, Test Acc: 0.6687
Epoch 160, Loss: 0.0002, Test Acc: 0.6668
Epoch 180, Loss: 0.0002, Test Acc: 0.6654
Epoch 200, Loss: 0.0001, Test Acc: 0.6654
"""

# =============================================
# 7. Extract Top-N Link Prediction Results
# =============================================

# Compute node embeddings from the trained GCN model
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient tracking (for efficiency)
    z = model(x, train_pos_edge_index)  # Generate embeddings for each node

# Compute prediction scores for all test edges
# These scores indicate how likely a link exists between each node pair
pred_scores = decode(z, test_data.pos_edge_label_index.to(device))

# Select the top-N predicted links based on the highest scores
# In this case, we choose the top 10
topk = torch.topk(pred_scores, k=10)

# Move the indices to CPU for further processing (if model was on GPU)
top_indices = topk.indices.cpu()

# Retrieve the corresponding node pairs (edges) from the test set
# Shape: [2, 10] → each column represents a (source, target) node pair
top_edge_indices = test_data.pos_edge_label_index[:, top_indices]

# Move the predicted scores of the top edges to CPU
top_scores = topk.values.cpu()

# Format and display the top predicted links
import pandas as pd
from IPython.display import display

# Convert the top edge tensor into a NumPy array of shape [10, 2]
top_edges_np = top_edge_indices.T.numpy()

# Create a DataFrame to neatly display the results
df_top_links = pd.DataFrame({
    'Source Node': top_edges_np[:, 0],        # First node of the edge
    'Target Node': top_edges_np[:, 1],        # Second node of the edge
    'Predicted Score': top_scores.numpy()     # Link strength (dot product)
})

# Display the resulting DataFrame in notebook
display(df_top_links)

# ================================
# 8. Top 10 predicted links
# ================================
# top_edge_indices shape: [2, 10]
top_edges_set = set(tuple(sorted(edge)) for edge in top_edge_indices.T.cpu().numpy())

# All ground-truth positive test edges
real_edges_set = set(tuple(sorted(edge)) for edge in test_data.pos_edge_label_index.T.cpu().numpy())

# Compare: Does each predicted link actually exist in the test set?
results = []
for edge in top_edges_set:
    exists = edge in real_edges_set
    results.append({'Source Node': edge[0], 'Target Node': edge[1], 'Actually Exists': exists})

# Create a DataFrame for display
df_existence_check = pd.DataFrame(results)
display(df_existence_check)

"""
		‌Source Node 	Target Node 	Actually Exists
0 	1093 	2367 	True
1 	1154 	1358 	True
2 	2075 	2667 	True
3 	831 	1011 	True
4 	99 	  2455 	True
5 	2289 	2464 	True
6 	446 	1507 	True
7 	2075 	2668 	True
8 	1136 	2359 	True
9 	998 	1431 	True
"""

# ================================
# 9. Visualize Graph
# ================================
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph and add the predicted edges
G = nx.Graph()
G.add_edges_from(predicted_edges)

# Create a subgraph that only contains nodes involved in the predictions
H = G.subgraph(set(sum(predicted_edges, ())))  # flatten node list

# Generate layout (spring_layout is stable and fast)
pos = nx.spring_layout(H, seed=42)

# Plot settings
plt.figure(figsize=(8, 6))
nx.draw(
    H, pos,
    with_labels=True,
    node_color='#ffc0cb',
    node_size=700,
    edge_color='black',
    width=2
)
plt.title("Top 10 Predicted Links", fontsize=14)
plt.axis("off")
plt.tight_layout()

# Show the plot
plt.show()
