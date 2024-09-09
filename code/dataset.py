import numpy as np
import string
from typing import List
import os
import argparse
#import networkx as nx

class Node(object):
    def __init__(self, node_id, node_type):
        self.id = node_id
        self.type = node_type
        self.prev = []
        self.next = []
    def __repr__(self):
        return f'{self.type}{self.id}'

class Nback(object):
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

class Tree(object):
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

        self.vocab = list(np.arange(self.vocab_size))

        # Create the transition matrix
        rng = np.random.default_rng(seed=seed)
        self.transition = {self.vocab[i]:np.array([max_prob if j==target else (1-max_prob)/(vocab_size-1)
                                                   for j in range(vocab_size)])
                                                   for i,target in enumerate(rng.permutation(vocab_size))}

    def sample_graph(self, seq_len:int, rng:np.random.Generator):
        '''
        Generate a binary-branching tree structure with a given sequence length.
        '''
        num_tokens = 1
        root = Node(0,'a')
        terms = [root]
        nodes = [root]
        while num_tokens < seq_len:
            node = terms.pop(0)
            if rng.random()>0.7 and num_tokens <= seq_len-5: # bifurcation
                b_node = Node(num_tokens,'b') # confusing due to 0 indexing
                b_node.prev = [node]
                node.next = [b_node]
                nodes.append(b_node)

                child_1 = Node(num_tokens+1,'a') # confusing due to 0 indexing
                child_2 = Node(num_tokens+2,'a') # confusing due to 0 indexing
                child_1.prev = [b_node]
                child_2.prev = [b_node]
                b_node.next = [child_1, child_2]
                nodes.extend([child_1, child_2])

                grandchild_1 = Node(num_tokens+3,'a')
                grandchild_2 = Node(num_tokens+4,'a')
                grandchild_1.prev = [child_1]
                grandchild_2.prev = [child_2]
                child_1.next = [grandchild_1]
                child_2.next = [grandchild_2]
                nodes.extend([grandchild_1, grandchild_2])

                terms.extend([grandchild_1, grandchild_2])
                num_tokens += 5
            else:
                next_node = Node(num_tokens,'a')
                next_node.prev = [node]
                node.next = [next_node]
                nodes.append(next_node)

                terms.append(next_node)
                num_tokens += 1
        return nodes, root

    def sample_dfs(self, root: Node, rng: np.random.Generator):
        stack = [root]
        nodes = []
        sent = []
        while len(stack) > 0:
            node = stack.pop(-1) # remove the last node in the stack
            node.id = len(sent) # reset the node id to the position in the sentence
            nodes.append(node)

            # sample token
            if len(node.prev)==0:
                token = rng.choice(self.vocab)
                token_type = 'a'
            else:
                probs = self.transition[node.prev[0].value]
                token = rng.choice(self.vocab, p=probs)
                if len(node.next)==0:
                    node.type = 'c'
                token_type = node.type

            node.value = token
            sent.append(token_type+str(token))

            # update stack
            for child in node.next:
                stack.append(child)
        return nodes, sent

    def calc_cross_entropy(self, root: Node, seq_len:int, rng: np.random.Generator, term_prob:float):
        stack = [root]
        nodes = []
        sent = []
        cross_entropy = 0
        while len(stack) > 0:
            node = stack.pop(-1) # remove the last node in the stack
            node.id = len(sent) # reset the node id to the position in the sentence
            nodes.append(node)

            # sample token
            if len(node.prev)==0:
                token = rng.choice(self.vocab)
                cross_entropy -= np.log(1/self.vocab_size)
                token_type = 'a'
            else:
                probs = self.transition[node.prev[0].value]
                token = rng.choice(self.vocab, p=probs)
                if len(node.next)==0:
                    node.type = 'c'
                token_type = node.type

                if node.prev[0].type in ['b','c']:
                    # if the preceding token is b or c, then the following token is always a
                    assert node.type=='a'
                    scl = 1.0
                elif len(node.prev[0].prev)>0:
                    if node.prev[0].prev[0].type in ['b','c']:
                        # if the token second before is b or c, then the following token is either a or c
                        assert node.type in ['a','c']
                        scl = term_prob if node.type=='c' else (1-term_prob)
                    else:
                        if node.type=='b':
                            scl = 0.3
                        else:
                            scl = 0.7*term_prob if node.type=='c' else 0.7*(1-term_prob)
                elif len(nodes) > seq_len-5:
                    # if the remaining tokens are less than 5, then the following token is either a or c
                    assert node.type in ['a','c']
                    scl = term_prob if node.type=='c' else (1-term_prob)
                else:
                    if node.type=='b':
                        scl = 0.3
                    else:
                        scl = 0.7*term_prob if node.type=='c' else 0.7*(1-term_prob)

                neg_log_prob = -np.log(probs[token]*scl)
                cross_entropy += neg_log_prob
            node.value = token
            sent.append(token_type+str(token))

            # update stack
            for child in node.next:
                stack.append(child)
        return cross_entropy

    def convert_to_mat(self, nodes: List[Node]):
        ids = np.array([int(node.__repr__()[1:]) for node in nodes])
        assert np.all(ids == np.arange(len(ids)))
        mat = []
        for node in nodes:
            mat.append([1 if len(node.prev)>0 and j==node.prev[0].id else 0 for j in range(len(ids))])
        mat = np.array(mat)
        return mat

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

    if args.graph_type == 'nback':
        generator = Nback(args.graph_type, args.vocab_size, args.max_prob, args.seed)
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

        trn_samples = []
        val_samples = []
        tst_samples = []
        for n in range(1,6):
            dirname = f'nback-{n}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
            with open(f'{args.base_dir}/dataset/{dirname}/trn.txt', 'r') as f:
                loaded_samples = f.read().split('\n')
            trn_samples.append(loaded_samples)

            with open(f'{args.base_dir}/dataset/{dirname}/val.txt', 'r') as f:
                loaded_samples = f.read().split('\n')
            val_samples.append(loaded_samples)

            with open(f'{args.base_dir}/dataset/{dirname}/tst.txt', 'r') as f:
                loaded_samples = f.read().split('\n')
            tst_samples.append(loaded_samples)

        trn_samples = list(np.array(trn_samples).T.reshape(5*10000))
        val_samples = list(np.array(val_samples).T.reshape(5*100))
        tst_samples = list(np.array(tst_samples).T.reshape(5*100))

        dirname = f'nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
        os.makedirs(f'{args.base_dir}/dataset/{dirname}',exist_ok=True)
        with open(f'{args.base_dir}/dataset/{dirname}/trn.txt', 'w') as f:
            f.write('\n'.join(trn_samples))
        with open(f'{args.base_dir}/dataset/{dirname}/val.txt', 'w') as f:
            f.write('\n'.join(val_samples))
        with open(f'{args.base_dir}/dataset/{dirname}/tst.txt', 'w') as f:
            f.write('\n'.join(tst_samples))

    elif args.graph_type == 'tree':
        generator = Tree(args.graph_type, args.vocab_size, args.max_prob, args.seed)
        rng = np.random.default_rng(seed=args.seed)

        # Generate 120 distinct structures
        roots = []
        names = []
        num_graphs = 0
        while num_graphs < 1120:
            nodes, root = generator.sample_graph(args.seq_len, rng)
            name = ' '.join([node.__repr__() for node in nodes])
            if name not in names:
                roots.append(root)
                names.append(name)
                num_graphs += 1
        assert len(roots) == 1120
        print(f"Generated {len(roots)} graphs")

        trn_mats = []
        trn_samples = []
        val_samples = []
        tst_samples = []
        for graph_id, root in enumerate(roots[:100]):
            print(f"Generating sentences for {graph_id}")
            # generate a sentence for calculating the template matrix
            nodes, _ = generator.sample_dfs(root, rng)
            mat = generator.convert_to_mat(nodes)
            trn_mats.append(mat)

            # generate sentences
            sents = [' '.join(generator.sample_dfs(root, rng)[1]) for _ in range(20000)]
            sents = list(set(sents))
            assert len(sents) > 10200, "not enough sentences"

            # choose sentences for the tree-all condition (1000-10-10 split)
            trn_samples.append(sents[:1000])
            val_samples.append(sents[10000:10010])
            tst_samples.append(sents[10100:10110])

            # save complete trn, val and tst sets for a single graph (10000-100-100 split)
            dirname = f'tree-{graph_id}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
            os.makedirs(f'{args.base_dir}/dataset/{dirname}',exist_ok=True)
            np.save(f'{args.base_dir}/dataset/{dirname}/mat.npy', mat)
            with open(f'{args.base_dir}/dataset/{dirname}/trn.txt', 'w') as f:
                f.write('\n'.join(sents[:10000]))
            with open(f'{args.base_dir}/dataset/{dirname}/val.txt', 'w') as f:
                f.write('\n'.join(sents[10000:10100]))
            with open(f'{args.base_dir}/dataset/{dirname}/tst.txt', 'w') as f:
                f.write('\n'.join(sents[10100:10200]))

        # save complete tree-all condition (10000-1000-1000 split)
        trn_samples = list(np.array(trn_samples).T.reshape(100*1000))
        val_samples = list(np.array(val_samples[:10]).T.reshape(10*10))
        tst_samples = list(np.array(tst_samples[:10]).T.reshape(10*10))

        dirname = f'tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
        os.makedirs(f'{args.base_dir}/dataset/{dirname}', exist_ok=True)
        np.save(f'{args.base_dir}/dataset/{dirname}/trn_mat.npy', np.array(trn_mats))
        with open(f'{args.base_dir}/dataset/{dirname}/trn.txt', 'w') as f:
            f.write('\n'.join(trn_samples))
        with open(f'{args.base_dir}/dataset/{dirname}/val.txt', 'w') as f:
            f.write('\n'.join(val_samples))
        with open(f'{args.base_dir}/dataset/{dirname}/tst.txt', 'w') as f:
            f.write('\n'.join(tst_samples))

        # val samples from ood graphs
        ex_val_samples = []
        ex_val_mats = []
        for graph_id, root in enumerate(roots[100:110]):
            # generate a sentence for calculating the template matrix
            nodes, _ = generator.sample_dfs(root, rng)
            mat = generator.convert_to_mat(nodes)
            ex_val_mats.append(mat)

            # generate sentences
            sents = [' '.join(generator.sample_dfs(root, rng)[1]) for _ in range(25)]
            sents = list(set(sents))
            assert len(sents) > 10, "not enough sentences"
            ex_val_samples.append(sents[:10])

        ex_val_samples = list(np.array(ex_val_samples).T.reshape(10*10))
        np.save(f'{args.base_dir}/dataset/{dirname}/ex_val_mat.npy', np.array(ex_val_mats))
        with open(f'{args.base_dir}/dataset/{dirname}/ex_val.txt', 'w') as f:
            f.write('\n'.join(ex_val_samples))

        # tst samples from ood graphs
        ex_tst_samples = []
        ex_tst_mats = []
        for graph_id, root in enumerate(roots[110:120]):
            # generate a sentence for calculating the template matrix
            nodes, _ = generator.sample_dfs(root, rng)
            mat = generator.convert_to_mat(nodes)
            ex_tst_mats.append(mat)

            # generate sentences
            sents = [' '.join(generator.sample_dfs(root, rng)[1]) for _ in range(25)]
            sents = list(set(sents))
            assert len(sents) > 10, "not enough sentences"
            ex_tst_samples.append(sents[:10])

        ex_tst_samples = list(np.array(ex_tst_samples).T.reshape(10*10))
        np.save(f'{args.base_dir}/dataset/{dirname}/ex_tst_mat.npy', np.array(ex_tst_mats))
        with open(f'{args.base_dir}/dataset/{dirname}/ex_tst.txt', 'w') as f:
            f.write('\n'.join(ex_tst_samples))

        # ood graphs for non-parametric estimation of attention disribution
        temp_samples = []
        temp_mats = []
        for graph_id, root in enumerate(roots[120:]):
            # generate a sentence for calculating the template matrix
            nodes, _ = generator.sample_dfs(root, rng)
            mat = generator.convert_to_mat(nodes)
            temp_mats.append(mat)

            # generate sentences
            sents = [' '.join(generator.sample_dfs(root, rng)[1]) for _ in range(25)]
            sents = list(set(sents))
            assert len(sents) > 10, "not enough sentences"
            temp_samples.append(sents[:10])

        temp_samples = list(np.array(temp_samples).T.reshape(1000*10))
        np.save(f'{args.base_dir}/dataset/{dirname}/templates_mat.npy', np.array(temp_mats))
        with open(f'{args.base_dir}/dataset/{dirname}/templates.txt', 'w') as f:
            f.write('\n'.join(temp_samples))