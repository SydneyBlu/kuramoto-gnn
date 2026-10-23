import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.contrib.nn.models import KuramotoGNN
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--coupling', type=float, default=1.0)
parser.add_argument('--time', type=float, default=8.0,
                    help='Integration time, i.e., the continuous depth')
parser.add_argument('--method', type=str, default='rk4',
                    help="'euler', 'rk4', or any 'torchdiffeq' method")
parser.add_argument('--step_size', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')

init_wandb(name=f'KuramotoGNN-{args.dataset}', time=args.time,
           coupling=args.coupling, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                'Planetoid')
# A sparse adjacency matrix lets `KuramotoConv` evaluate the coupling term via
# a single sparse-dense matrix product per solver stage:
transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToSparseTensor(layout=torch.sparse_csr),
])
dataset = Planetoid(path, args.dataset, transform=transform)
data = dataset[0].to(device)

model = KuramotoGNN(
    dataset.num_features,
    args.hidden_channels,
    dataset.num_classes,
    coupling=args.coupling,
    time=args.time,
    input_dropout=0.5,
    dropout=0.5,
    method=args.method,
    step_size=args.step_size,
    heads=args.heads,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.adj_t).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


times = []
best_val_acc = test_acc = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)
print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')
