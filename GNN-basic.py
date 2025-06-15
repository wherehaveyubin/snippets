import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

## Step 1: Create demo data (5 nodes, 6 edges)
# 1. Node features (e.g., population, average income, building density, temperature)
#    - Temperature includes extreme heat (e.g., ≥ 35°C)
x = torch.tensor([
    [1000, 40000, 0.3, 36],   # A (extreme heat)
    [2500, 30000, 0.5, 33],   # B (normal)
    [1800, 70000, 0.6, 37],   # C (extreme heat)
    [3000, 50000, 0.4, 31],   # D (normal)
    [1200, 65000, 0.2, 38],   # E (extreme heat)
], dtype=torch.float)

# 2. Labels (mobility decrease: 1 = decreased, 0 = maintained/increased)
y = torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)

# 3. edge_index: specify edges bidirectionally
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]
], dtype=torch.long)

# 4. edge_attr: distance (km), mobility frequency
edge_attr = torch.tensor([
    [2.5, 130],  # A-B
    [2.5, 130],  # B-A
    [3.0, 90],   # B-C
    [3.0, 90],   # C-B
    [1.8, 60],   # C-D
    [1.8, 60],   # D-C
    [2.2, 50],   # D-E
    [2.2, 50],   # E-D
    [4.0, 40],   # E-A
    [4.0, 40],   # A-E
], dtype=torch.float)

# 5. Build the PyG Data object
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
# Data(x=[5, 4], edge_index=[2, 10], edge_attr=[10, 2], y=[5])

## Step 2: Define basic GCN model
# For now, this model does not use edge_attr; only node features are used
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_channels=4, out_channels=8)
        self.conv2 = GCNConv(in_channels=8, out_channels=2)  # Binary classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

## Step 3: Training loop
model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1, 101):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        pred = out.argmax(dim=1)
        acc = int((pred == data.y).sum()) / data.num_nodes
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.2f}')
"""
Epoch 010, Loss: 483.8182, Acc: 0.40
Epoch 020, Loss: 252.6908, Acc: 0.60
Epoch 030, Loss: 204.8049, Acc: 0.40
Epoch 040, Loss: 105.9340, Acc: 0.40
Epoch 050, Loss: 70.4427, Acc: 0.40
Epoch 060, Loss: 17.4059, Acc: 0.60
Epoch 070, Loss: 7.2339, Acc: 0.40
Epoch 080, Loss: 11.0529, Acc: 0.60
Epoch 090, Loss: 18.2249, Acc: 0.60
Epoch 100, Loss: 15.2985, Acc: 0.40
"""
