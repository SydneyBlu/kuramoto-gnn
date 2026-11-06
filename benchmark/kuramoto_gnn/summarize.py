"""Aggregates the per-run lines printed by `compare_official.py` shards.

Example:
    python summarize.py results/*.log
"""
import re
import sys

import numpy as np

PATTERN = re.compile(
    r'Run (\d+) \(split seed\s*(\d+)\): official ([\d.]+) \((\d+)s\), '
    r'PyG \(official options\) ([\d.]+) \((\d+)s\), '
    r'PyG \(paper equation\) ([\d.]+) \((\d+)s\), '
    r'max train-loss diff official/PyG ([\d.e+-]+)')
NAMES = ['Official', 'PyG (official options)', 'PyG (paper equation)']

runs = {}
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            match = PATTERN.match(line)
            if match is not None:
                g = match.groups()
                runs[int(g[0])] = dict(
                    split_seed=int(g[1]),
                    accs=[float(g[2]), float(g[4]),
                          float(g[6])],
                    times=[int(g[3]), int(g[5]),
                           int(g[7])],
                    loss_diff=float(g[8]),
                )

print('| Run | Split seed | ' + ' | '.join(NAMES) + ' | Max train-loss diff |')
print('| --- | --- | ' + ' | '.join(['---'] * len(NAMES)) + ' | --- |')
for run, r in sorted(runs.items()):
    accs = ' | '.join(f'{100 * acc:.2f}' for acc in r['accs'])
    print(f"| {run} | {r['split_seed']} | {accs} | {r['loss_diff']:.1e} |")

accs = 100 * np.array([r['accs'] for r in runs.values()])
times = np.array([r['times'] for r in runs.values()])
print(f'\n| Model | Test accuracy ({len(runs)} runs) | Median train time |')
print('| --- | --- | --- |')
for i, name in enumerate(NAMES):
    print(f'| {name} | {accs[:, i].mean():.2f} ± {accs[:, i].std():.2f} | '
          f'{np.median(times[:, i]):.0f}s |')

diff = np.abs(accs[:, 1] - accs[:, 0])
print(
    f'\nPer-run |official - PyG (official options)|: mean {diff.mean():.2f}, '
    f'max {diff.max():.2f} points')
