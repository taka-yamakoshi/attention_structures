import numpy as np
import random
import torch

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gen_dataset_name(args):
    return f'{args.graph_type}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'

def gen_run_name(args):
    model_stat = f'{args.num_layers}_{args.num_heads}_{args.hidden_size}_{args.intermediate_size}'
    run_stat = f'{args.bias}_{args.beta}_{args.datasize}_{args.batchsize_trn}_{args.batchsize_val}_{args.lr}_{args.scheduler_type}_{args.num_epochs}_{args.run_seed}'
    return f'{args.model_type}_{gen_dataset_name(args)}_{model_stat}_{run_stat}'