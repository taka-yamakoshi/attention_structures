import numpy as np
import string
import os
import argparse
import networkx as nx

class SentenceGenerator(object):
    def __init__(self, graph_type: str, vocab_size: int, max_prob: float, seed:int=1234):
        '''
        graph_type: type of structure (e.g. nback)
        vocab_size: size of vocabulary
        max_prob: probability assigned to the preferred transition
        seed: random seed for the transition matrix
        '''
        self.graph_type = graph_type
        self.vocab_size = vocab_size
        self.max_prob = max_prob

        assert vocab_size < 27, "vocab_size is too large"
        self.vocab = list(string.ascii_lowercase[:vocab_size])

        # Create the transition matrix
        rng = np.random.default_rng(seed=seed)
        self.transition = {self.vocab[i]:np.array([max_prob if j==target else (1-max_prob)/(vocab_size-1)
                                                   for j in range(vocab_size)])
                                                   for i,target in enumerate(rng.permutation(vocab_size))}
    
    def sample(self, mat:np.ndarray, seq_len:int, rng:np.random.Generator):
        sources = np.arange(seq_len)[mat.sum(axis=1)==0]
        graph = nx.from_numpy_array(mat.T, create_using=nx.DiGraph)
        out = np.array([' ' for _ in range(seq_len)])
        for source in sources:
            assert out[source]==' '
            out[source] = rng.choice(self.vocab)
            for edge in nx.dfs_edges(graph,source=source):
                assert out[edge[1]]==' '
                out[edge[1]] = rng.choice(self.vocab, p=self.transition[out[edge[0]]])
        assert np.all(out!=' ')
        return out

    def generate_nback(self, n:int, seq_len:int, nsamples:int, seed:int=1234):
        mat = np.array([[1 if j==i-n else 0
                         for j in range(seq_len)]
                         for i in range(seq_len)])
        rng = np.random.default_rng(seed=seed)
        samples = np.array([self.sample(mat, seq_len, rng) for _ in range(nsamples)])
        return np.hstack([np.array([[str(n)]]*nsamples), samples])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_type', type = str, required = True, choices = ['nback'])
    parser.add_argument('--vocab_size', type = int, default = 5)
    parser.add_argument('--max_prob', type=float, default = 0.8)
    parser.add_argument('--seq_len', type = int, default = 16)
    parser.add_argument('--seed', type = int, default = 1234)
    args = parser.parse_args()
    print(f'running with {args}')
    args.base_dir = os.environ.get("MY_DATA_PATH")

    if args.graph_type == 'nback':
        generator = SentenceGenerator(args.graph_type, args.vocab_size, args.max_prob, args.seed)
        for n in range(1,6):
            dirname = f'nback-{n}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
            os.makedirs(f'{args.base_dir}/dataset/{dirname}',exist_ok=True)

            samples = generator.generate_nback(n=n, seq_len=args.seq_len, nsamples=20000, seed=args.seed)
            print("w/ overlap",len(samples))
            samples = list(set([' '.join(sample) for sample in samples]))
            assert len(samples)>10200, "not enough samples"
            print("w/out overlap",len(samples))
            with open(f'{args.base_dir}/dataset/{dirname}/trn.txt', 'w') as f:
                f.write('\n'.join(samples[:10000]))
            with open(f'{args.base_dir}/dataset/{dirname}/val.txt', 'w') as f:
                f.write('\n'.join(samples[10000:10100]))
            with open(f'{args.base_dir}/dataset/{dirname}/tst.txt', 'w') as f:
                f.write('\n'.join(samples[10100:10200]))