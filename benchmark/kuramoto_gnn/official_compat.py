"""Makes the official KuramotoGNN code importable on current PyTorch and PyG.

The official code (https://github.com/Fsoft-AIC/Reducing-Over-smoothing-via-a-
Kuramoto-Model-based-Approach) targets `torch==1.8.1` and
`torch-geometric==1.7.0`, and imports `torch_sparse` and `torch_scatter`,
which have no wheels for current PyTorch releases. The functions it calls on
the Cora path are re-implemented below via `torch_geometric.utils.scatter`,
following the reference implementations of both packages. Modules that are
only needed for graph rewiring and OGB datasets are replaced by stubs that
raise if they are ever called.
"""
import importlib
import sys
import types

import torch_geometric.datasets  # noqa: F401
import torch_geometric.nn  # noqa: F401
import torch_geometric.transforms  # noqa: F401
from torch_geometric.utils import scatter as pyg_scatter


def _spmm(index, value, m, n, matrix):
    # `torch_sparse.spmm`: out[row] += value * matrix[col].
    row, col = index[0], index[1]
    out = matrix.index_select(-2, col) * value.unsqueeze(-1)
    return pyg_scatter(out, row, dim=-2, dim_size=m, reduce='sum')


def _scatter(src, index, dim=-1, out=None, dim_size=None, reduce='sum'):
    assert out is None
    return pyg_scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)


def _scatter_add(src, index, dim=-1, out=None, dim_size=None):
    return _scatter(src, index, dim, out, dim_size, reduce='sum')


def _unsupported(name):
    def func(*args, **kwargs):
        raise NotImplementedError(f"'{name}' is not available in this "
                                  f"compatibility layer")

    return func


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def install(official_src: str) -> None:
    r"""Installs the shims and adds :obj:`official_src` to the import path.
    Needs to be called after :obj:`torch_geometric` has been imported, such
    that PyG itself does not pick up the shims.
    """
    try:
        importlib.import_module('torch_sparse')
    except ImportError:
        sys.modules['torch_sparse'] = _module(
            'torch_sparse', spmm=_spmm,
            coalesce=_unsupported('torch_sparse.coalesce'))
    try:
        importlib.import_module('torch_scatter')
    except ImportError:
        sys.modules['torch_scatter'] = _module(
            'torch_scatter', scatter=_scatter, scatter_add=_scatter_add,
            segment_csr=_unsupported('torch_scatter.segment_csr'),
            gather_csr=_unsupported('torch_scatter.gather_csr'))
    try:
        importlib.import_module('ogb.nodeproppred')
    except ImportError:
        sys.modules['ogb'] = _module('ogb')
        sys.modules['ogb.nodeproppred'] = _module(
            'ogb.nodeproppred',
            PygNodePropPredDataset=_unsupported('PygNodePropPredDataset'))

    # The official `graph_rewiring` module needs `pykeops` and `numba`:
    sys.modules['graph_rewiring'] = _module(
        'graph_rewiring', **{
            name: _unsupported(f'graph_rewiring.{name}')
            for name in [
                'get_two_hop', 'apply_gdc', 'make_symmetric',
                'apply_pos_dist_rewire', 'apply_KNN', 'apply_beltrami',
                'apply_edge_sampling'
            ]
        })

    sys.path.insert(0, official_src)
