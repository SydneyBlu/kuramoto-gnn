"""Compares `KuramotoGNN` against the official KuramotoGNN implementation.

1. Numerics: with copied weights, compares logits, gradients and the number
   of ODE function evaluations of both models.
2. Training: trains both models from the same initialization on the same
   splits with identical dropout masks, following the official training loop
   (`src/run_GNN.py`), and reports test accuracies.

Example:
    python compare_official.py --official_src \
        ~/Reducing-Over-smoothing-via-a-Kuramoto-Model-based-Approach/src
"""
import argparse
import os.path as osp
import time

import numpy as np
import official_compat
import torch
import torch.nn.functional as F

from torch_geometric.contrib.nn.models import KuramotoGNN

parser = argparse.ArgumentParser()
parser.add_argument('--official_src', type=str, required=True,
                    help="Path to the official repository's 'src' directory")
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--time', type=float, default=8.0)
parser.add_argument('--coupling', type=float, default=0.9)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument(
    '--first_run', type=int, default=0,
    help='Index of the first run, to shard runs over processes')
parser.add_argument('--threads', type=int, default=None)
parser.add_argument('--skip_numerics', action='store_true')
parser.add_argument('--epochs', type=int, default=None,
                    help='Defaults to the official best parameters')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument(
    '--data_root', type=str,
    default=osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'data',
                     'official'))
args = parser.parse_args()

if args.threads is not None:
    torch.set_num_threads(args.threads)
official_compat.install(osp.expanduser(args.official_src))
from best_params import best_params_dict  # noqa: E402
from GNN import GNN  # noqa: E402

from data import get_dataset, set_train_val_test_split  # noqa: E402


def make_opt():
    # Mirrors the overrides in `main()` of the official `run_GNN.py`:
    opt = dict(best_params_dict[args.dataset])
    opt.update(time=args.time, method='dopri5', step_size=1.0,
               hidden_dim=args.hidden_dim, add_source=False,
               coupling_strength=args.coupling, kuramoto=1, is_webKB=False,
               split_rate=20, beltrami=False)
    assert opt['block'] == 'attention' and opt['function'] == 'laplacian'
    assert opt['attention_type'] == 'scaled_dot' and not opt['use_mlp']
    assert not opt['batch_norm'] and not opt['fc_out']
    return opt


def make_pyg_model(opt, dataset, paper_equation=False):
    kwargs = dict(
        normalization='squareplus' if opt['square_plus'] else 'softmax',
        normalize_over='source' if opt['attention_norm_idx'] else 'target',
        mean_field=True,
        detach_omega=True,
        clamp=True,
    )
    if paper_equation:  # The Kuramoto equation of the paper:
        kwargs = dict(mean_field=False, detach_omega=False, clamp=False)

    return KuramotoGNN(
        dataset.num_features,
        opt['hidden_dim'],
        dataset.num_classes,
        coupling=opt['coupling_strength'],
        time=opt['time'],
        input_dropout=opt['input_dropout'],
        dropout=opt['dropout'],
        method=opt['method'],
        heads=opt['heads'],
        attention_channels=opt['attention_dim'],
        add_self_loops=opt['self_loop_weight'] > 0,
        adjoint=opt['adjoint'],
        solver_kwargs=dict(rtol=opt['tol_scale'] * 1e-9,
                           atol=opt['tol_scale'] * 1e-7),
        **kwargs,
    )


def copy_weights(official, model):
    model.encoder.load_state_dict(official.m1.state_dict())
    model.decoder.load_state_dict(official.m2.state_dict())
    att = official.odeblock.multihead_att_layer
    model.conv.lin_query.load_state_dict(att.Q.state_dict())
    model.conv.lin_key.load_state_dict(att.K.state_dict())


def param_pairs(official, model):
    att = official.odeblock.multihead_att_layer
    return [
        ('encoder', official.m1, model.encoder),
        ('decoder', official.m2, model.decoder),
        ('query', att.Q, model.conv.lin_query),
        ('key', att.K, model.conv.lin_key),
    ]


def check_numerics(official, model, data, edge_index):
    nfe = [0]
    model.conv.register_propagate_forward_hook(
        lambda *_: nfe.__setitem__(0, nfe[0] + 1))

    def forward_backward(module, forward):
        # Backpropagate right away, such that only one autograd graph through
        # the solver is alive at a time:
        module.eval()
        module.zero_grad()
        out = forward()
        F.cross_entropy(out[data.train_mask],
                        data.y[data.train_mask]).backward()
        return out.detach()

    official.resetNFE()
    out1 = forward_backward(official, lambda: official(data.x)[0])
    nfe1 = official.getNFE()
    out2 = forward_backward(model, lambda: model(data.x, edge_index))
    nfe2 = nfe[0]

    print(f'Logits:  max abs diff {(out1 - out2).abs().max():.2e} '
          f'(max abs value {out1.abs().max():.2e})')
    print(f'NFE:     official {nfe1}, PyG {nfe2}')
    for name, lin1, lin2 in param_pairs(official, model):
        for attr in ['weight', 'bias']:
            g1 = getattr(lin1, attr).grad
            g2 = getattr(lin2, attr).grad
            print(f'Grad {name}.{attr}: max abs diff '
                  f'{(g1 - g2).abs().max():.2e} '
                  f'(max abs value {g1.abs().max():.2e})')


def get_optimizer(opt, model):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizers = dict(sgd=torch.optim.SGD, rmsprop=torch.optim.RMSprop,
                      adagrad=torch.optim.Adagrad, adam=torch.optim.Adam,
                      adamax=torch.optim.Adamax)
    return optimizers[opt['optimizer']](params, lr=opt['lr'],
                                        weight_decay=opt['decay'])


def train_and_test(model, forward, data, opt, epochs, seed):
    # Follows `train()`, `test()` and `main()` of the official `run_GNN.py`:
    optimizer = get_optimizer(opt, model)
    best_val_acc = test_acc = 0
    patience = 0
    losses = []
    for epoch in range(1, epochs):
        torch.manual_seed(seed * 100_000 + epoch)  # Same dropout masks.
        model.train()
        optimizer.zero_grad()
        out = forward(model)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        losses.append(float(loss))

        model.eval()
        with torch.no_grad():
            pred = forward(model).argmax(dim=-1)
        accs = [
            int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
            for mask in [data.val_mask, data.test_mask]
        ]
        if accs[0] > best_val_acc:
            best_val_acc, test_acc = accs
            patience = 0
        else:
            patience += 1
            if patience == 100:
                break
    return test_acc, losses


def main():
    opt = make_opt()
    epochs = args.epochs or opt['epoch']
    dataset = get_dataset(opt, args.data_root, opt['not_lcc'])
    data = dataset.data
    # The official code aggregates `out[edge_index[0]] += a * x[edge_index[1]]`
    # (target-to-source), while PyG aggregates from `edge_index[0]` into
    # `edge_index[1]`:
    edge_index = data.edge_index.flip([0])
    print(f'{args.dataset}: {data.num_nodes} nodes (LCC), '
          f'{data.edge_index.size(1)} edges, T={args.time}, '
          f'K={args.coupling}, hidden={args.hidden_dim}, epochs={epochs}')

    if not args.skip_numerics:
        torch.manual_seed(args.seed)
        official = GNN(opt, dataset)
        model = make_pyg_model(opt, dataset)
        copy_weights(official, model)
        print('\n== Numerics (shared weights, eval mode) ==')
        check_numerics(official, model, data, edge_index)

    print('\n== Training (shared init, splits and dropout masks) ==')
    names = ['official', 'PyG (official options)', 'PyG (paper equation)']
    results = {name: [] for name in names}
    loss_diffs = []
    for run in range(args.first_run, args.first_run + args.runs):
        split_seed = np.random.RandomState(args.seed + run).randint(0, 1000)
        data = set_train_val_test_split(split_seed, data, num_development=1500,
                                        num_per_class=opt['split_rate'])
        torch.manual_seed(args.seed + run)
        official = GNN(opt, dataset)
        models = [
            official,
            make_pyg_model(opt, dataset),
            make_pyg_model(opt, dataset, paper_equation=True),
        ]
        for model in models[1:]:
            copy_weights(official, model)

        def official_forward(model, data=data):
            # The official loop resets the counter checked against `max_nfe`:
            model.resetNFE()
            return model(data.x)[0]

        def pyg_forward(model, data=data):
            return model(data.x, edge_index)

        forwards = [official_forward, pyg_forward, pyg_forward]
        line, losses = [], []
        for name, model, forward in zip(names, models, forwards):
            start = time.perf_counter()
            acc, loss = train_and_test(model, forward, data, opt, epochs, run)
            results[name].append(acc)
            losses.append(loss)
            line.append(
                f'{name} {acc:.4f} ({time.perf_counter() - start:.0f}s)')
        loss_diffs.append(
            np.abs(np.array(losses[0]) - np.array(losses[1])).max())
        print(f'Run {run:02d} (split seed {split_seed:3d}): ' +
              ', '.join(line) +
              f', max train-loss diff official/PyG {loss_diffs[-1]:.1e}')

    print(f'\n== Test accuracy over {args.runs} runs ==')
    for name, accs in results.items():
        accs = 100 * np.array(accs)
        print(f'{name:24s} {accs.mean():.2f} ± {accs.std():.2f}')
    diffs = 100 * (np.array(results[names[1]]) - np.array(results[names[0]]))
    print(f'Per-run |official - PyG (official options)|: '
          f'max {np.abs(diffs).max():.2f} points')


if __name__ == '__main__':
    main()
