import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import random

# ---------------------------
# 1. Node feature generation
# ---------------------------
num_nodes = 10
sums = np.random.uniform(0.8, 0.9, size=num_nodes)
ratios = np.random.dirichlet(alpha=[2, 2, 2], size=num_nodes)  # Adjust alpha for distribution skew

pct_white = sums * ratios[:, 0]
pct_black = sums * ratios[:, 1]
pct_asian = sums * ratios[:, 2]

num_nodes = 10
node_features = pd.DataFrame({
    'ID': list(range(num_nodes)),
    'pct_white': pct_white,
    'pct_black': pct_black,
    'pct_asian': pct_asian,
    'median_income': np.random.randint(30000, 100000, num_nodes),
    'pct_highrise': np.random.rand(num_nodes),
    'pct_green': np.random.rand(num_nodes)
})

# Z-score normalization
node_features['median_income'] = (node_features['median_income'] - node_features['median_income'].mean()) / node_features['median_income'].std()

# ---------------------------
# 2. Edge list and attributes generation
# ---------------------------
edge_list = []
edge_attrs = []

for _ in range(30):  # generate 30 edges
    src = random.randint(0, num_nodes - 1)
    dst = random.randint(0, num_nodes - 1)
    hour = random.randint(0, 23)
    heat_index = round(random.uniform(70, 110), 2)
    weight = random.uniform(0, 1)  # prediction target

    edge_list.append([src, dst])
    edge_attrs.append([hour, heat_index, weight])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
edge_features = edge_attrs[:, :2]  # hour, heat_index
edge_weights = edge_attrs[:, 2]    # target variable

# ---------------------------
# 3. Create PyTorch Geometric Data object
# ---------------------------
x = torch.tensor(node_features.drop(columns='ID').values, dtype=torch.float)
data = Data(x=x, edge_index=edge_index, edge_attr=edge_features, y=edge_weights)

# ---------------------------
# 4. Define GCN + Edge MLP model
# ---------------------------
class EdgeWeightPredictor(torch.nn.Module):
    def __init__(self, in_channels, edge_feat_dim):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, 32)
        self.gcn2 = GCNConv(32, 16)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(16 * 2 + edge_feat_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.edge_mlp(edge_input).squeeze()

# ---------------------------
# 5. Train the model
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EdgeWeightPredictor(x.size(1), edge_features.size(1)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.MSELoss()

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = loss_fn(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ✅ Output example
# Check prediction results
pred = model(data.x, data.edge_index, data.edge_attr).detach().cpu().numpy()
true = data.y.cpu().numpy()

result_df = pd.DataFrame({
    "src": edge_index[0].cpu().numpy(),
    "dst": edge_index[1].cpu().numpy(),
    "hour": edge_features[:, 0].cpu().numpy(),
    "heat_index": edge_features[:, 1].cpu().numpy(),
    "predicted_weight": pred,
    "actual_weight": true
})

print(result_df.head())
