# KuramotoGNN: PyG vs. official implementation

Compares `torch_geometric.contrib.nn.models.KuramotoGNN` against the [official KuramotoGNN code](https://github.com/Fsoft-AIC/Reducing-Over-smoothing-via-a-Kuramoto-Model-based-Approach) on Cora.
This directory is not part of the upstream PR (it lives on the `kuramoto-gnn-comparison` branch).

## Setup

The official code targets `torch==1.8.1` / `torch-geometric==1.7.0`.
`official_compat.py` makes it importable on current PyTorch and PyG without modifying it:
`torch_sparse.spmm` and the `torch_scatter` functions on the Cora path are re-implemented via `torch_geometric.utils.scatter`, and modules only needed for graph rewiring and OGB (`pykeops`, `numba`, `ogb`) are stubbed out.

```bash
pip install torchdiffeq scikit-learn six
python compare_official.py --official_src /path/to/Reducing-Over-smoothing-via-a-Kuramoto-Model-based-Approach/src
```

Each process needs several GB of RAM, since both models backpropagate through all ~600 `dopri5` function evaluations of a forward pass.
Runs can be sharded over processes via `--first_run`/`--runs` (use `--skip_numerics` for all but one shard) if memory allows, and the shard logs merged via `python summarize.py results/*.log`.

The official model is configured exactly as `run_GNN.py --method dopri5 --no_early --dataset Cora` does:
the Cora entry of `best_params.py`, `hidden_dim=16`, 20 labels per class on the largest connected component, and `dopri5` with `atol = tol_scale * 1e-7`, `rtol = tol_scale * 1e-9`.
The PyG model reproduces it via

```python
KuramotoGNN(..., clamp=True, mean_field=True, detach_omega=True,
            normalization='squareplus', normalize_over='source',
            method='dopri5', heads=8, attention_channels=128, ...)
```

and is fed `edge_index.flip(0)`, since the official code aggregates from `edge_index[1]` into `edge_index[0]`.

## What is compared

1. **Numerics.** Both models share weights; logits, gradients of the training loss, and the number of ODE function evaluations (NFE) are compared in eval mode.
1. **Training.** For each run, both models start from the same initialization, use the same random split (as in `run_GNN.py`) and the same dropout masks, and follow the official training loop (Adamax, gradient clipping at 5, best-validation test accuracy, patience 100).
   A third model, `PyG (paper equation)`, uses the layer defaults instead: the exact Kuramoto coupling `sum_j a_ij sin(x_j - x_i)`, softmax attention, no clamp and no detached frequencies.

## Results

**Numerics** (shared weights, eval mode): logits agree to `1.5e-5` (max. absolute value `6.3`), gradients of the training loss w.r.t. all shared parameters agree to `1e-8`, and both models take exactly 578 ODE function evaluations, *i.e.*, `dopri5` takes identical steps.

**Training** (10 runs): pending.
