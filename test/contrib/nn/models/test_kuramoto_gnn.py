import math

import pytest
import torch

from torch_geometric.contrib.nn.models import KuramotoGNN
from torch_geometric.utils import to_torch_csc_tensor


@pytest.mark.parametrize('clamp', [False, True])
def test_kuramoto_gnn(clamp):
    x = torch.randn(6, 16)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
    ])
    adj = to_torch_csc_tensor(edge_index, size=(6, 6))

    model = KuramotoGNN(16, 8, 3, coupling=0.5, time=2.0, input_dropout=0.5,
                        dropout=0.1, clamp=clamp, heads=2)
    assert str(model) == ('KuramotoGNN(16, 8, 3, conv=KuramotoConv(8, '
                          "coupling=0.5, time=2.0, method='rk4', heads=2))")

    model.eval()
    out = model(x, edge_index)
    assert out.size() == (6, 3)
    assert torch.allclose(model(x, adj.t()), out, atol=1e-6)

    model.train()
    model(x, edge_index).sum().backward()
    for param in model.parameters():
        assert param.grad is not None

    model.reset_parameters()
    assert not torch.allclose(model.eval()(x, edge_index), out)


def test_kuramoto_gnn_clamp():
    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

    model = KuramotoGNN(16, 8, 3, clamp=True)
    phases = []
    model.conv.register_forward_pre_hook(
        lambda module, inputs: phases.append(inputs[0]))
    model(x, edge_index)
    assert phases[0].min() >= 0.0 and phases[0].max() <= math.pi


def test_kuramoto_gnn_edge_weight():
    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    edge_weight = torch.rand(edge_index.size(1))

    model = KuramotoGNN(16, 8, 3, attention=False)
    out1 = model(x, edge_index)
    out2 = model(x, edge_index, edge_weight)
    assert out1.size() == out2.size() == (6, 3)
    assert not torch.allclose(out1, out2)
