import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.index import index2ptr
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    index_sort,
    is_torch_sparse_tensor,
    scatter,
    softmax,
    spmm,
    to_edge_index,
)

# Methods that `torchdiffeq` integrates on a fixed grid of size `step_size`:
FIXED_GRID_METHODS = {
    'euler', 'midpoint', 'heun2', 'heun3', 'rk4', 'explicit_adams',
    'implicit_adams', 'fixed_adams'
}


def squareplus(src: Tensor, index: Tensor,
               num_nodes: Optional[int] = None) -> Tensor:
    r"""Normalizes :obj:`src` within the groups given by :obj:`index` via
    :math:`\textrm{squareplus}(x) = (x + \sqrt{x^2 + 4}) / 2` in place of
    :math:`\exp(x)`, as done in the official KuramotoGNN implementation.
    Scores are shifted by their global maximum beforehand.
    """
    if src.numel() == 0:
        return src
    out = src - src.max()
    out = (out + (out.square() + 4).sqrt()) / 2
    out_sum = scatter(out, index, dim=0, dim_size=num_nodes, reduce='sum')
    return out / (out_sum.index_select(0, index) + 1e-16)


def _euler_step(func: Callable[[Tensor], Tensor], x: Tensor,
                dt: float) -> Tensor:
    return x + dt * func(x)


def _rk4_step(func: Callable[[Tensor], Tensor], x: Tensor,
              dt: float) -> Tensor:
    # The 3/8-rule variant of RK4, matching `torchdiffeq`'s "rk4" method:
    k1 = func(x)
    k2 = func(x + dt * k1 / 3)
    k3 = func(x + dt * (k2 - k1 / 3))
    k4 = func(x + dt * (k1 - k2 + k3))
    return x + (k1 + 3 * (k2 + k3) + k4) * dt / 8


class KuramotoConv(MessagePassing):
    r"""The Kuramoto oscillator operator from the `"From Coupled Oscillators
    to Graph Neural Networks: Reducing Over-smoothing via a Kuramoto
    Model-based Approach" <https://arxiv.org/abs/2311.03260>`_ paper.

    Node features are treated as phases of coupled oscillators, which evolve
    according to the Kuramoto model

    .. math::
        \frac{d \mathbf{x}_i(t)}{dt} = \boldsymbol{\omega}_i + K
        \sum_{j \in \mathcal{N}(i) \cup \{ i \}} a_{i,j}
        \sin \left( \mathbf{x}_j(t) - \mathbf{x}_i(t) \right),

    which is integrated from :math:`t = 0` to :math:`t = T`, starting from
    the input features :math:`\mathbf{X}(0)`, to return :math:`\mathbf{X}(T)`.
    By default, the natural frequencies are given by
    :math:`\boldsymbol{\Omega} = \mathbf{X}(0)`.
    Different natural frequencies let nodes synchronize in frequency rather
    than in phase, which reduces over-smoothing for long integration times.

    By default, the coupling coefficients :math:`a_{i,j}` are given by
    scaled dot-product attention, computed once from :math:`\mathbf{X}(0)`
    and averaged over :math:`H` heads:

    .. math::
        a_{i,j} = \frac{1}{H} \sum_{h=1}^H \textrm{softmax}_j \left(
        \frac{(\mathbf{W}^{(h)}_Q \mathbf{x}_i(0))^{\top}
        (\mathbf{W}^{(h)}_K \mathbf{x}_j(0))}{\sqrt{d}} \right),

    where :math:`d` denotes the number of attention channels per head.
    If :obj:`attention=False`, :math:`a_{i,j}` is instead given by the edge
    weights (or one if no edge weights are given), which recovers the
    classical Kuramoto model on a (weighted) graph.

    For sparse adjacency inputs, each evaluation of the vector field reduces
    to a single sparse-dense matrix product
    :math:`\mathbf{A} [\sin(\mathbf{X}) \, \Vert \, \cos(\mathbf{X})]` via
    :math:`\sin(\mathbf{x}_j - \mathbf{x}_i) = \sin(\mathbf{x}_j)
    \cos(\mathbf{x}_i) - \cos(\mathbf{x}_j) \sin(\mathbf{x}_i)`, which avoids
    materializing per-edge messages at every solver stage and thus reduces
    memory consumption during training.
    Note that backpropagating to the values of :class:`torch.sparse` matrices
    can be slow on CPUs, such that :obj:`edge_index` or
    :class:`torch_sparse.SparseTensor` inputs may train faster there.

    If :obj:`mean_field=True`, the coupling term is replaced by the
    mean-field formulation of the official implementation

    .. math::
        \frac{d \mathbf{x}_i(t)}{dt} = \boldsymbol{\omega}_i + K
        \mathbf{r}_i \odot \sin(\boldsymbol{\phi}_i - \mathbf{x}_i),
        \qquad \mathbf{r}_i = \Big| \sum_j a_{i,j} e^{\iota \mathbf{x}_j}
        \Big|, \quad \boldsymbol{\phi}_i = \sum_j a_{i,j} \mathbf{x}_j,

    which uses the averaged phase :math:`\boldsymbol{\phi}_i` in place of the
    argument of :math:`\sum_j a_{i,j} e^{\iota \mathbf{x}_j}`, and is thus
    not identical to the equation above. Unlike the equation above, it is not
    differentiable where :math:`\sum_j a_{i,j} e^{\iota \mathbf{x}_j}`
    vanishes.

    .. note::

        The `official implementation
        <https://github.com/Fsoft-AIC/Reducing-Over-smoothing-via-a-Kuramoto-
        Model-based-Approach>`_ always uses :obj:`mean_field=True` and
        :obj:`detach_omega=True`, and clamps :math:`\mathbf{X}(0)` to
        :math:`[0, \pi]` (see :obj:`clamp` in
        :class:`~torch_geometric.contrib.nn.models.KuramotoGNN`).
        The attention normalization and direction are dataset-specific
        hyperparameters there, *e.g.*, :obj:`normalization="squareplus"` and
        :obj:`normalize_over="source"` on Cora.
        It further initializes the query and key weights to :math:`10^{-5}`,
        such that attention coefficients start out (almost) uniform.

    .. note::

        For an example of using :class:`KuramotoConv`, see
        `examples/contrib/kuramoto_gnn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/kuramoto_gnn.py>`_.

    Args:
        in_channels (int): Size of each input sample.
        coupling (float, optional): The coupling strength :math:`K`.
            (default: :obj:`1.0`)
        time (float, optional): The integration time :math:`T`, *i.e.*, the
            continuous depth of the operator. (default: :obj:`1.0`)
        method (str, optional): The ODE solver. :obj:`"euler"` and
            :obj:`"rk4"` use built-in fixed-step solvers; any other method,
            *e.g.*, :obj:`"dopri5"` (as used in the paper), is forwarded to
            :obj:`torchdiffeq.odeint`. (default: :obj:`"rk4"`)
        step_size (float, optional): The step size of fixed-step solvers.
            The last step is shortened to end exactly at :obj:`time`.
            (default: :obj:`1.0`)
        attention (bool, optional): If set to :obj:`False`, will use edge
            weights instead of learned attention coefficients. If set to
            :obj:`True`, edge weights are not supported and the values of
            sparse input matrices are ignored. (default: :obj:`True`)
        heads (int, optional): Number of attention heads :math:`H`.
            (default: :obj:`1`)
        attention_channels (int, optional): Size of the query and key
            projections, summed over all heads. Needs to be divisible by
            :obj:`heads`. If set to :obj:`None`, will use :obj:`in_channels`.
            (default: :obj:`None`)
        normalization (str, optional): The attention normalization,
            :obj:`"softmax"` or :obj:`"squareplus"`. (default:
            :obj:`"softmax"`)
        normalize_over (str, optional): Whether attention coefficients sum to
            one over the incoming edges of each target node
            (:obj:`"target"`) or over the outgoing edges of each source node
            (:obj:`"source"`). (default: :obj:`"target"`)
        mean_field (bool, optional): If set to :obj:`True`, will use the
            mean-field formulation of the official implementation.
            (default: :obj:`False`)
        detach_omega (bool, optional): If set to :obj:`True`, gradients will
            not flow through the natural frequencies
            :math:`\boldsymbol{\Omega}`, as in the official implementation.
            (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. Self-loops added via this option
            are assigned an edge weight of one. (default: :obj:`True`)
        adjoint (bool, optional): If set to :obj:`True`, will backpropagate
            via the adjoint method of :obj:`torchdiffeq.odeint_adjoint`,
            which uses memory independent of the number of solver steps.
            (default: :obj:`False`)
        solver_kwargs (Dict[str, Any], optional): Additional arguments passed
            to :obj:`torchdiffeq.odeint` or :obj:`torchdiffeq.odeint_adjoint`,
            *e.g.*, :obj:`rtol`, :obj:`atol` or :obj:`options`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`
          or sparse matrix :math:`(|\mathcal{V}|, |\mathcal{V}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*,
          natural frequencies :math:`(|\mathcal{V}|, F)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    def __init__(
        self,
        in_channels: int,
        coupling: float = 1.0,
        time: float = 1.0,
        method: str = 'rk4',
        step_size: float = 1.0,
        attention: bool = True,
        heads: int = 1,
        attention_channels: Optional[int] = None,
        normalization: str = 'softmax',
        normalize_over: str = 'target',
        mean_field: bool = False,
        detach_omega: bool = False,
        add_self_loops: bool = True,
        adjoint: bool = False,
        solver_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if time <= 0:
            raise ValueError(f"'time' must be positive (got {time})")
        if step_size <= 0:
            raise ValueError(f"'step_size' must be positive (got {step_size})")
        if normalization not in {'softmax', 'squareplus'}:
            raise ValueError(f"'normalization' must be 'softmax' or "
                             f"'squareplus' (got '{normalization}')")
        if normalize_over not in {'target', 'source'}:
            raise ValueError(f"'normalize_over' must be 'target' or 'source' "
                             f"(got '{normalize_over}')")

        attention_channels = attention_channels or in_channels
        if attention and attention_channels % heads != 0:
            raise ValueError(f"'attention_channels' ({attention_channels}) "
                             f"must be divisible by 'heads' ({heads})")

        self.in_channels = in_channels
        self.coupling = coupling
        self.time = time
        self.method = method
        self.step_size = step_size
        self.attention = attention
        self.heads = heads
        self.attention_channels = attention_channels
        self.normalization = normalization
        self.normalize_over = normalize_over
        self.mean_field = mean_field
        self.detach_omega = detach_omega
        self.add_self_loops = add_self_loops
        self.adjoint = adjoint
        self.solver_kwargs = solver_kwargs or {}

        if attention:
            self.lin_query = Linear(in_channels, attention_channels)
            self.lin_key = Linear(in_channels, attention_channels)
        else:
            self.register_module('lin_query', None)
            self.register_module('lin_key', None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if self.lin_query is not None:
            self.lin_query.reset_parameters()
        if self.lin_key is not None:
            self.lin_key.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        omega: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The initial phases :math:`\mathbf{X}(0)`.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights, which are
                only supported if :obj:`attention=False`.
                (default: :obj:`None`)
            omega (torch.Tensor, optional): The natural frequencies
                :math:`\boldsymbol{\Omega}`. If set to :obj:`None`, will use
                :math:`\mathbf{X}(0)`. (default: :obj:`None`)
        """
        if self.attention and edge_weight is not None:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"'edge_weight' when 'attention=True'. Set "
                             f"'attention=False' to use edge weights as "
                             f"coupling coefficients")

        num_nodes = x.size(0)
        omega = x if omega is None else omega
        if self.detach_omega:
            omega = omega.detach()

        # Self-loops and coupling coefficients are computed on the edge list.
        # Sparse inputs are converted back afterwards, such that each
        # evaluation of the vector field is a single sparse matrix product:
        is_sparse_tensor = isinstance(edge_index, SparseTensor)
        is_torch_sparse = is_torch_sparse_tensor(edge_index)
        if is_sparse_tensor:
            row, col, edge_weight = edge_index.coo()
            edge_index = torch.stack([col, row], dim=0)
        elif is_torch_sparse and edge_index.layout == torch.sparse_coo:
            # `to_edge_index` marks COO matrices as coalesced in-place, which
            # does not support backpropagation to their values:
            coo = edge_index.coalesce()
            edge_index, edge_weight = coo.indices().flip([0]), coo.values()
        elif is_torch_sparse:
            edge_index, edge_weight = to_edge_index(edge_index)
            edge_index = edge_index.flip([0])

        if self.attention:  # Values of sparse inputs are ignored.
            edge_weight = None

        if self.add_self_loops:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)

        if self.attention:
            assert self.lin_query is not None and self.lin_key is not None
            H, C = self.heads, self.attention_channels // self.heads
            query = self.lin_query(x).view(-1, H, C)
            key = self.lin_key(x).view(-1, H, C)
            # edge_updater_type: (query: Tensor, key: Tensor)
            alpha = self.edge_updater(edge_index, query=query, key=key)
        elif edge_weight is None:
            alpha = x.new_ones(edge_index.size(1))
        else:
            alpha = edge_weight

        adj: Adj = edge_index
        edge_weight = alpha
        csr: Optional[Tuple[Tensor, Tensor, Tensor]] = None
        if is_sparse_tensor:
            adj = SparseTensor(row=edge_index[1], col=edge_index[0],
                               value=alpha,
                               sparse_sizes=(num_nodes, num_nodes))
            edge_weight = None
        elif is_torch_sparse:
            # Sort the edges by target node once to obtain the CSR layout of
            # the transposed adjacency matrix:
            row, col = edge_index[1], edge_index[0]
            _, perm = index_sort(row * num_nodes + col,
                                 max_value=num_nodes * num_nodes)
            csr = (index2ptr(row[perm], num_nodes), col[perm], alpha[perm])
            edge_weight = None

        if self.adjoint:
            # The adjoint method differentiates the vector field w.r.t. each
            # of `alpha` and `omega` separately. Since `alpha` is computed
            # from `omega = x` by default, `omega` needs to enter the vector
            # field as a separate node to not count the path through `alpha`
            # twice:
            omega = omega.view_as(omega)

        def vector_field(x: Tensor) -> Tensor:
            adj_t = adj
            if csr is not None:
                # A new sparse matrix is created for every evaluation, since
                # backpropagating through a single one used at every solver
                # stage sums up sparse gradients, which PyTorch does not
                # support on CPUs without MKL (*e.g.*, on Apple silicon):
                adj_t = torch.sparse_csr_tensor(*csr,
                                                size=(num_nodes, num_nodes))
            return omega + self.coupling * self._coupling(
                x, adj_t, edge_weight)

        adjoint_params = tuple(t for t in (alpha, omega) if t.requires_grad)
        return self._integrate(vector_field, x, adjoint_params)

    def _coupling(self, x: Tensor, adj: Adj, edge_weight: OptTensor) -> Tensor:
        if self.mean_field:
            h = torch.cat([x.sin(), x.cos(), x], dim=-1)
            # propagate_type: (x: OptTensor, h: OptTensor,
            #                  edge_weight: OptTensor)
            out = self.propagate(adj, x=None, h=h, edge_weight=edge_weight)
            sin_aggr, cos_aggr, x_aggr = out.split(x.size(-1), dim=-1)
            r = (sin_aggr.square() + cos_aggr.square()).clamp_min(1e-12).sqrt()
            return r * torch.sin(x_aggr - x)

        # propagate_type: (x: OptTensor, h: OptTensor, edge_weight: OptTensor)
        return self.propagate(adj, x=x, h=None, edge_weight=edge_weight)

    def _integrate(
        self,
        func: Callable[[Tensor], Tensor],
        x: Tensor,
        adjoint_params: Tuple[Tensor, ...],
    ) -> Tensor:
        if self.method in {'euler', 'rk4'} and not self.adjoint:
            step = _euler_step if self.method == 'euler' else _rk4_step
            num_steps = max(1, math.ceil(round(self.time / self.step_size, 8)))
            for _ in range(num_steps - 1):
                x = step(func, x, self.step_size)
            return step(func, x, self.time - (num_steps - 1) * self.step_size)

        try:
            from torchdiffeq import odeint, odeint_adjoint
        except ImportError as e:
            raise ImportError(f"'{self.__class__.__name__}' requires the "
                              f"'torchdiffeq' package for method "
                              f"'{self.method}' or 'adjoint=True'") from e

        kwargs = dict(self.solver_kwargs)
        if self.method in FIXED_GRID_METHODS:
            options = dict(kwargs.get('options') or {})
            options.setdefault('step_size', self.step_size)
            kwargs['options'] = options

        def ode_func(t: Tensor, x: Tensor) -> Tensor:
            return func(x)

        t = torch.tensor([0.0, self.time], dtype=x.dtype, device=x.device)
        if self.adjoint:
            out = odeint_adjoint(ode_func, x, t, method=self.method,
                                 adjoint_params=adjoint_params, **kwargs)
        else:
            out = odeint(ode_func, x, t, method=self.method, **kwargs)
        return out[-1]

    def edge_update(self, query_i: Tensor, key_j: Tensor, index: Tensor,
                    edge_index_j: Tensor, dim_size: Optional[int],
                    size_j: Optional[int]) -> Tensor:
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(query_i.size(-1))

        if self.normalize_over == 'source':
            index, dim_size = edge_index_j, size_j

        if self.normalization == 'softmax':
            alpha = softmax(alpha, index, num_nodes=dim_size)
        else:
            alpha = squareplus(alpha, index, num_nodes=dim_size)

        return alpha.mean(dim=-1)

    def message(self, x_i: OptTensor, x_j: OptTensor, h_j: OptTensor,
                edge_weight: OptTensor) -> Tensor:
        if h_j is not None:  # Mean-field, `h = [sin(X) || cos(X) || X]`.
            out = h_j
        else:
            assert x_i is not None and x_j is not None
            out = torch.sin(x_j - x_i)
        return out if edge_weight is None else edge_weight.view(-1, 1) * out

    def message_and_aggregate(self, adj_t: Adj, x: OptTensor,
                              h: OptTensor) -> Tensor:
        if h is not None:  # Mean-field, `h = [sin(X) || cos(X) || X]`.
            return spmm(adj_t, h)

        assert x is not None
        sin_x, cos_x = x.sin(), x.cos()
        out = spmm(adj_t, torch.cat([sin_x, cos_x], dim=-1))
        sin_aggr, cos_aggr = out.split(x.size(-1), dim=-1)
        return sin_aggr * cos_x - cos_aggr * sin_x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'coupling={self.coupling}, time={self.time}, '
                f'method={self.method!r}, heads={self.heads})')
