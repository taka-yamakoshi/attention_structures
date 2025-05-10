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

def calc_attns_l2_loss(args, attns, pretrained_attns):
    return torch.mean(torch.sum((attns-shuffle_attns_all(pretrained_attns, args))**2,dim=(2,3,4)))

def calc_logits_kl_loss(args, logprobs, pretrained_logprobs, attn_mask):
    assert len(logprobs.shape)==3 and len(pretrained_logprobs.shape)==3
    if args.topk is not None:
        bsize, seqlen, _ = pretrained_logprobs.shape
        topk_indices = torch.argsort(pretrained_logprobs.to(torch.float16), dim=-1, descending=True)[:,:,:args.topk]
        logprobs = logprobs[torch.arange(bsize)[...,None,None],torch.arange(seqlen)[None,...,None],topk_indices]
        pretrained_logprobs = pretrained_logprobs[torch.arange(bsize)[...,None,None],torch.arange(seqlen)[None,...,None],topk_indices]
    attn_mask = torch.nn.functional.pad(attn_mask, (0, 1), value=0)
    shift_attn_mask = attn_mask[..., 1:].contiguous()
    kldiv = 0.0
    for mask, lgprb, prt_lgprb in zip(shift_attn_mask, logprobs, pretrained_logprobs):
        kldiv += torch.mean(torch.sum(torch.exp(prt_lgprb[mask==1])*(-lgprb[mask==1]),dim=-1))
    return kldiv/len(shift_attn_mask)