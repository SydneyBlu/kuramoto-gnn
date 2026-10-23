import math
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.contrib.nn.conv import KuramotoConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptTensor


class KuramotoGNN(torch.nn.Module):
    r"""The KuramotoGNN model from the `"From Coupled Oscillators to Graph
    Neural Networks: Reducing Over-smoothing via a Kuramoto Model-based
    Approach" <https://arxiv.org/abs/2311.03260>`_ paper.

    A linear encoder maps node features to initial phases
    :math:`\mathbf{X}(0)`, which also serve as natural frequencies
    :math:`\boldsymbol{\Omega} = \mathbf{X}(0)`.
    The phases are then evolved by a single
    :class:`~torch_geometric.contrib.nn.conv.KuramotoConv` up to time
    :math:`T`, and decoded by a linear layer:

    .. math::
        \mathbf{X}(0) &= \mathbf{X} \mathbf{W}_{\textrm{in}} +
        \mathbf{b}_{\textrm{in}}

        \mathbf{X}(T) &= \textrm{KuramotoConv}(\mathbf{X}(0))

        \mathbf{Y} &= \sigma(\mathbf{X}(T)) \mathbf{W}_{\textrm{out}} +
        \mathbf{b}_{\textrm{out}}

    .. note::

        For an example of using :class:`KuramotoGNN`, see
        `examples/contrib/kuramoto_gnn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/kuramoto_gnn.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each oscillator state.
        out_channels (int): Size of each output sample.
        coupling (float, optional): The coupling strength :math:`K`.
            (default: :obj:`1.0`)
        time (float, optional): The integration time :math:`T`.
            (default: :obj:`1.0`)
        input_dropout (float, optional): Dropout probability of the input
            features. (default: :obj:`0.0`)
        dropout (float, optional): Dropout probability of the final oscillator
            states. (default: :obj:`0.0`)
        act (str or Callable, optional): The non-linear activation function
            :math:`\sigma` applied to the final oscillator states.
            (default: :obj:`"relu"`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        clamp (bool, optional): If set to :obj:`True`, will clamp the initial
            phases :math:`\mathbf{X}(0)` to :math:`[0, \pi]`, as in the
            official implementation. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`~torch_geometric.contrib.nn.conv.KuramotoConv`.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        coupling: float = 1.0,
        time: float = 1.0,
        input_dropout: float = 0.0,
        dropout: float = 0.0,
        act: Union[str, Callable, None] = 'relu',
        act_kwargs: Optional[Dict[str, Any]] = None,
        clamp: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.clamp = clamp

        self.encoder = Linear(in_channels, hidden_channels)
        self.conv = KuramotoConv(hidden_channels, coupling=coupling, time=time,
                                 **kwargs)
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.decoder = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.encoder.reset_parameters()
        self.conv.reset_parameters()
        self.decoder.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights, which are
                only supported if :obj:`attention=False`.
                (default: :obj:`None`)
        """
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.encoder(x)
        if self.clamp:
            x = x.clamp(min=0.0, max=math.pi)
        x = self.conv(x, edge_index, edge_weight)
        if self.act is not None:
            x = self.act(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.decoder(x)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'conv={self.conv})')
