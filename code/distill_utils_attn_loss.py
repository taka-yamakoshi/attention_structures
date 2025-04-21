import torch

def shuffle_attns(attns, args, dim):
    assert len(attns.shape)==5 and attns.shape[0]==args.num_layers and attns.shape[2]==args.num_heads
    idx = torch.randperm(attns.shape[dim], device=args.device)
    return torch.index_select(attns, dim, idx)

def shuffle_attns_all(attns, args):
    if args.shuffle is None:
        return attns
    if 'head' in args.shuffle.split('-'):
        attns = shuffle_attns(attns, args, dim=2)
    if 'layer' in args.shuffle.split('-'):
        attns = shuffle_attns(attns, args, dim=0)
    if 'batch' in args.shuffle.split('-'):
        attns = shuffle_attns(attns, args, dim=1)
    return attns