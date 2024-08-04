import numpy as np
from typing import List
import os
import argparse

from dataset import Tree

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_type', type = str, required = True, choices = ['nback','tree'])
    parser.add_argument('--vocab_size', type = int, default = 5)
    parser.add_argument('--max_prob', type=float, default = 0.8)
    parser.add_argument('--seq_len', type = int, default = 16)
    parser.add_argument('--seed', type = int, default = 1234)
    args = parser.parse_args()
    print(f'running with {args}')
    args.base_dir = os.environ.get("MY_DATA_PATH")

    generator = Tree(args.graph_type, args.vocab_size, args.max_prob, args.seed)
    rng = np.random.default_rng(seed=args.seed)

    # Generate 100 distinct structures
    roots = []
    names = []
    num_graphs = 0
    while num_graphs < 1000:
        nodes, root = generator.sample_graph(args.seq_len, rng)
        name = ' '.join([node.__repr__() for node in nodes])
        if name not in names:
            roots.append(root)
            names.append(name)
            num_graphs += 1
    assert len(roots) == 1000
    print(f"Generated {len(roots)} graphs")

    dirname = f'tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
    os.makedirs(f'{args.base_dir}/cross_entropy/{dirname}', exist_ok=True)
    for term_prob in np.linspace(0.2,0.8,7):
        print(f'Running {term_prob:.1g}')
        cross_entropy = np.empty((1000,100))
        for graph_id, root in enumerate(roots):
            cross_entropy[graph_id] = [generator.calc_cross_entropy(root, args.seq_len, rng, term_prob) for _ in range(100)]
        np.save(f'{args.base_dir}/cross_entropy/{dirname}/term_prob_{term_prob:.1g}.npy', cross_entropy)