import numpy as np
import torch
import argparse
import os
import glob
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from distill_utils import seed_everything
from distill_utils_eval import evaluate_linzen_agg, evaluate_blimp, evaluate_zorro

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type = str, default=None)
    parser.add_argument('--max_length', type = int, default=None)
    parser.add_argument('--run_seed', type = int, default=1234)
    parser.add_argument('--core_id', type = int, default = 0)
    parser.add_argument('--version', type = str, required=True)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(f'../distill_results/{args.version}', exist_ok=True)

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

    model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/distill_models/{args.version}/{args.model_name}/best')
    model.eval()
    model.to(args.device)

    blimp_prefix = '../blimp/data'
    blimp_tasks = [file.replace(blimp_prefix+'/','').replace('.jsonl','') for file in glob.glob(f'{blimp_prefix}/*.jsonl')]
    zorro_prefix = '../Zorro/sentences/babyberta'
    zorro_tasks = [file.replace(zorro_prefix+'/','').replace('.txt','') for file in glob.glob(f'{zorro_prefix}/*.txt')]
    out_linzen = evaluate_linzen_agg(model, args)
    out_blimp = evaluate_blimp(model, args, blimp_tasks)
    out_zorro = evaluate_zorro(model, args, zorro_tasks)

    tasks = []
    perfs = []
    for key, val in out_linzen.items():
        tasks.append(key)
        perfs.append(val)
    for key, val in out_blimp.items():
        tasks.append(key)
        perfs.append(val)
    for key, val in out_zorro.items():
        tasks.append(key)
        perfs.append(val)
    df = pd.DataFrame.from_dict({'tasks': tasks, 'acc': perfs})
    df.to_csv(f'../distill_results/{args.version}/{args.model_name}.csv', index=False)