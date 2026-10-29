import math

import pytest
import torch

import torch_geometric.typing
from torch_geometric.contrib.nn.conv import KuramotoConv
from torch_geometric.testing import onlyFullTest, onlyLinux, withPackage
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    to_dense_adj,
    to_torch_csc_tensor,
    to_torch_csr_tensor,
)


def dense_attention(conv, x, edge_index):
    # Dense reference of the coupling matrix `A[i, j] = a_ij`:
    N, H = x.size(0), conv.heads
    C = conv.attention_channels // H
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=N)
    mask = to_dense_adj(edge_index, max_num_nodes=N)[0].t().bool()

    query = conv.lin_query(x).view(N, H, C)
    key = conv.lin_key(x).view(N, H, C)
    score = torch.einsum('ihc,jhc->hij', query, key) / math.sqrt(C)

    dim = -1 if conv.normalize_over == 'target' else -2
    if conv.normalization == 'softmax':
        score = score.masked_fill(~mask, float('-inf'))
        return score.softmax(dim=dim).mean(dim=0)

    score = score - score[:, mask].max()
    score = (score + (score**2 + 4).sqrt()) / 2
    score = score.masked_fill(~mask, 0.0)
    return (score / score.sum(dim=dim, keepdim=True)).mean(dim=0)


def dense_vector_field(x, omega, A, coupling, mean_field=False):
    if mean_field:
        s, c, phi = A @ x.sin(), A @ x.cos(), A @ x
        return omega + coupling * (s**2 + c**2).sqrt() * (phi - x).sin()
    diff = x.unsqueeze(0) - x.unsqueeze(1)  # diff[i, j] = x_j - x_i
    return omega + coupling * (A.unsqueeze(-1) * diff.sin()).sum(dim=1)


def dense_integrate(func, x, time, num_steps, method):
    dt = time / num_steps
    for _ in range(num_steps):
        if method == 'euler':
            x = x + dt * func(x)
        else:  # 3/8-rule RK4:
            k1 = func(x)
            k2 = func(x + dt * k1 / 3)
            k3 = func(x + dt * (k2 - k1 / 3))
            k4 = func(x + dt * (k1 - k2 + k3))
            x = x + (k1 + 3 * (k2 + k3) + k4) * dt / 8
    return x


def get_graph():
    x = torch.randn(6, 8)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 5, 0],
    ])
    return x, edge_index


def test_kuramoto_conv():
    x, edge_index = get_graph()
    adj1 = to_torch_csc_tensor(edge_index, size=(6, 6))
    adj2 = to_torch_csr_tensor(edge_index.flip([0]), size=(6, 6))

    conv = KuramotoConv(8, time=2.0, step_size=0.5, heads=2)
    assert str(conv) == ('KuramotoConv(8, coupling=1.0, time=2.0, '
                         "method='rk4', heads=2)")

    out = conv(x, edge_index)
    assert out.size() == (6, 8)
    assert torch.allclose(conv(x, adj1.t()), out, atol=1e-6)
    assert torch.allclose(conv(x, adj2), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj3 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(6, 6))
        assert torch.allclose(conv(x, adj3.t()), out, atol=1e-6)

    with pytest.raises(ValueError, match="does not support 'edge_weight'"):
        conv(x, edge_index, torch.rand(edge_index.size(1)))


@pytest.mark.parametrize('method', ['euler', 'rk4'])
@pytest.mark.parametrize('mean_field', [False, True])
@pytest.mark.parametrize('normalization', ['softmax', 'squareplus'])
@pytest.mark.parametrize('normalize_over', ['target', 'source'])
def test_kuramoto_conv_dense_reference(method, mean_field, normalization,
                                       normalize_over):
    x, edge_index = get_graph()
    conv = KuramotoConv(8, coupling=0.8, time=1.5, method=method,
                        step_size=0.25, heads=2, attention_channels=12,
                        normalization=normalization,
                        normalize_over=normalize_over, mean_field=mean_field)

    A = dense_attention(conv, x, edge_index)
    expected = dense_integrate(
        lambda h: dense_vector_field(h, x, A, 0.8, mean_field),
        x,
        time=1.5,
        num_steps=6,
        method=method,
    )
    assert torch.allclose(conv(x, edge_index), expected, atol=1e-5)


def test_kuramoto_conv_edge_weight():
    x, edge_index = get_graph()
    edge_weight = torch.rand(edge_index.size(1))
    adj = to_torch_csc_tensor(edge_index, edge_weight, size=(6, 6))

    conv = KuramotoConv(8, coupling=0.5, time=1.0, step_size=0.1,
                        attention=False)
    assert conv.lin_query is None and conv.lin_key is None

    out = conv(x, edge_index, edge_weight)
    assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)

    if torch_geometric.typing.WITH_TORCH_SPARSE:
        adj = SparseTensor.from_edge_index(edge_index, edge_weight, (6, 6))
        assert torch.allclose(conv(x, adj.t()), out, atol=1e-6)

    # Classical Kuramoto model with coupling matrix `A[i, j] = w_ji`:
    A = to_dense_adj(edge_index, edge_attr=edge_weight)[0].t()
    expected = dense_integrate(lambda h: dense_vector_field(h, x, A, 0.5), x,
                               time=1.0, num_steps=10, method='rk4')
    assert torch.allclose(out, expected, atol=1e-5)

    # Without edge weights, all coupling coefficients equal one:
    out = conv(x, edge_index)
    expected = conv(x, edge_index, torch.ones(edge_index.size(1)))
    assert torch.allclose(out, expected, atol=1e-6)


def test_kuramoto_conv_uncoupled():
    x, edge_index = get_graph()
    omega = torch.randn_like(x)

    # Without coupling, phases rotate with their natural frequencies:
    conv = KuramotoConv(8, coupling=0.0, time=2.5, step_size=1.0)
    assert torch.allclose(conv(x, edge_index), x + 2.5 * x, atol=1e-6)
    out = conv(x, edge_index, omega=omega)
    assert torch.allclose(out, x + 2.5 * omega, atol=1e-6)


def test_kuramoto_conv_frequency_synchronization():
    # Strongly coupled oscillators lock to their mean natural frequency:
    x = torch.rand(4, 1)
    omega = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])

    conv1 = KuramotoConv(1, coupling=5.0, time=20.0, step_size=0.1,
                         attention=False)
    conv2 = KuramotoConv(1, coupling=5.0, time=21.0, step_size=0.1,
                         attention=False)
    x1 = conv1(x, edge_index, omega=omega)
    x2 = conv2(x, edge_index, omega=omega)
    assert torch.allclose(x2 - x1, omega.mean().expand_as(x1), atol=1e-3)
    assert (x2 - omega.mean() * 21.0).std() > 0.01  # Phases stay distinct.


def test_kuramoto_conv_gradients():
    x, edge_index = get_graph()
    x.requires_grad_()

    conv = KuramotoConv(8, time=2.0, step_size=0.5, heads=2)
    conv(x, edge_index).square().sum().backward()
    for grad in [x.grad, conv.lin_query.weight.grad, conv.lin_key.weight.grad]:
        assert grad is not None
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0

    # Gradients through the natural frequencies can be disabled:
    conv = KuramotoConv(8, coupling=0.0, time=2.0, attention=False)
    x.grad = None
    conv(x, edge_index).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 3.0))

    conv = KuramotoConv(8, coupling=0.0, time=2.0, attention=False,
                        detach_omega=True)
    x.grad = None
    conv(x, edge_index).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_kuramoto_conv_last_step():
    x, edge_index = get_graph()
    conv1 = KuramotoConv(8, time=1.25, method='euler', step_size=0.5)
    conv2 = KuramotoConv(8, time=1.25, method='euler', step_size=0.25)
    conv2.load_state_dict(conv1.state_dict())

    # Two steps of size 0.5 followed by one step of size 0.25:
    out = conv1(x, edge_index)
    conv1.time = 1.0
    out1 = conv1(x, edge_index)
    conv2.time = 1.0
    conv2.step_size = 0.5
    assert torch.allclose(conv2(x, edge_index), out1)
    conv2.time = 1.25
    assert not torch.allclose(conv2(x, edge_index), out1)
    assert torch.allclose(conv2(x, edge_index), out)


def test_kuramoto_conv_invalid_arguments():
    with pytest.raises(ValueError, match="must be divisible by 'heads'"):
        KuramotoConv(8, heads=3)
    with pytest.raises(ValueError, match="'normalization' must be"):
        KuramotoConv(8, normalization='sparsemax')
    with pytest.raises(ValueError, match="'normalize_over' must be"):
        KuramotoConv(8, normalize_over='both')
    with pytest.raises(ValueError, match="'time' must be positive"):
        KuramotoConv(8, time=0.0)


@withPackage('torchdiffeq')
@pytest.mark.parametrize('method', ['euler', 'rk4'])
def test_kuramoto_conv_torchdiffeq_fixed_step(method):
    from torchdiffeq import odeint

    x, edge_index = get_graph()
    conv = KuramotoConv(8, time=1.3, method=method, step_size=0.25, heads=2)
    out = conv(x, edge_index)

    # The built-in solvers use the same grid and schemes as `torchdiffeq`:
    A = dense_attention(conv, x, edge_index)
    expected = odeint(lambda t, h: dense_vector_field(h, x, A, 1.0), x,
                      torch.tensor([0.0, 1.3]), method=method,
                      options=dict(step_size=0.25))[-1]
    assert torch.allclose(out, expected, atol=1e-5)


@withPackage('torchdiffeq')
@pytest.mark.parametrize('mean_field', [False, True])
def test_kuramoto_conv_torchdiffeq_adaptive(mean_field):
    _, edge_index = get_graph()
    # Phases within one radian keep the mean-field `r_i` away from zero, where
    # it is not differentiable and solvers converge slowly:
    x = torch.rand(6, 8, dtype=torch.double, requires_grad=True)
    conv1 = KuramotoConv(8, time=2.0, method='rk4', step_size=0.01, heads=2,
                         mean_field=mean_field)
    conv2 = KuramotoConv(8, time=2.0, method='dopri5', heads=2,
                         mean_field=mean_field,
                         solver_kwargs=dict(rtol=1e-10, atol=1e-12))
    conv3 = KuramotoConv(8, time=2.0, method='dopri5', heads=2,
                         mean_field=mean_field, adjoint=True,
                         solver_kwargs=dict(rtol=1e-10, atol=1e-12))
    conv1.double()
    conv2.double().load_state_dict(conv1.state_dict())
    conv3.double().load_state_dict(conv1.state_dict())

    outs, grads = [], []
    for conv in [conv1, conv2, conv3]:
        out = conv(x, edge_index)
        x_grad, weight_grad = torch.autograd.grad(out.square().sum(),
                                                  [x, conv.lin_query.weight])
        outs.append(out)
        grads.append((x_grad, weight_grad))

    # Direct and adjoint backpropagation through an adaptive solver agree with
    # backpropagation through a fine fixed-step solver:
    for out, (x_grad, weight_grad) in zip(outs[1:], grads[1:]):
        assert torch.allclose(out, outs[0], atol=1e-6)
        assert torch.allclose(x_grad, grads[0][0], atol=1e-5)
        assert torch.allclose(weight_grad, grads[0][1], atol=1e-5)


@withPackage('torchdiffeq')
def test_kuramoto_conv_adjoint_sparse():
    x, edge_index = get_graph()
    adj = to_torch_csr_tensor(edge_index.flip([0]), size=(6, 6))

    conv = KuramotoConv(8, time=1.0, method='rk4', step_size=0.1, adjoint=True)
    out = conv(x, adj)
    assert torch.allclose(out, conv(x, edge_index), atol=1e-6)

    out.sum().backward()
    assert conv.lin_query.weight.grad is not None
    assert conv.lin_query.weight.grad.abs().sum() > 0


@onlyLinux
@onlyFullTest
@withPackage('torch>=2.1.0')
def test_compile_kuramoto_conv():
    x, edge_index = get_graph()
    conv = KuramotoConv(8, time=2.0, step_size=0.5, heads=2)

    out = torch.compile(conv)(x, edge_index)
    assert torch.allclose(conv(x, edge_index), out, atol=1e-6)
