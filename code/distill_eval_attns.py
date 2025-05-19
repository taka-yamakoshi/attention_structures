import numpy as np
import torch
import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from distill_utils import seed_everything
from distill_utils_eval import evaluate_linzen_attns

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default=None)
    parser.add_argument('--max_length', type = int, default=None)
    parser.add_argument('--num_samples', type = int, default=100)
    parser.add_argument('--run_seed', type = int, default=1234)
    parser.add_argument('--core_id', type = int, default = 0)
    parser.add_argument('--version', type = str, required=True)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    args.attns_out_dir = f"{args.base_dir}/distill_attns/{args.version}/{args.model_name}"
    os.makedirs(args.attns_out_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Fix the seed
    seed_everything(args.run_seed)
    args.rng = np.random.default_rng(args.run_seed)

    # Load the tokenizer
    if args.model_name.startswith('gpt2'):
        args.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
    elif args.model_name.startswith('llama2'):
        args.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                        cache_dir=args.cache_dir, token=os.environ.get('HF_TOKEN'))
    else:
        raise NotImplementedError
    args.tokenizer.pad_token = args.tokenizer.eos_token

    if args.model_name in ['gpt2']:
        model = AutoModelForCausalLM.from_pretrained(f'{args.model_name}',cache_dir=args.cache_dir,token=os.environ.get('HF_TOKEN'))
    else:
        model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/distill_models/{args.version}/{args.model_name}/best')
    model.eval()
    model.to(args.device)

    attns = evaluate_linzen_attns(model, args, num_samples=args.num_samples)
    np.save(f"{args.attns_out_dir}/attns.npy", attns)