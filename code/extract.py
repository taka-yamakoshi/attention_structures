import numpy as np
import torch
import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

from utils import gen_dataset_name, gen_run_name, seed_everything
from utils_dataset import load_sentences

def get_template_loaders(args):
    # Load the dataset and the data collator
    dataset = load_sentences(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=False)
    if args.graph_type.startswith('tree'):
        subset = 'temps'
    else:
        dataset['val'] = dataset['val'].shuffle(seed=args.run_seed)
        subset = 'val'
    data_loader = torch.utils.data.DataLoader(dataset[subset], batch_size=args.batchsize_val,collate_fn=data_collator)
    return dataset, data_loader

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type = str, default=None)
    parser.add_argument('--max_length', type = int, default=None)
    parser.add_argument('--dataset_name', type = str, default=None)
    parser.add_argument('--model_type', type = str)
    parser.add_argument('--graph_type', type = str, choices = ['tree-all', 'babylm', 'wiki'])
    parser.add_argument('--vocab_size', type = int, default = 5)
    parser.add_argument('--max_prob', type=float, default = 0.8)
    parser.add_argument('--seq_len', type = int, default = 16)
    parser.add_argument('--seed', type = int, default = 1234)
    
    parser.add_argument('--num_layers', type = int, default = 1)
    parser.add_argument('--num_heads', type = int, default = 1)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--intermediate_size', type = int, default = 512)

    parser.add_argument('--datasize', type = int, default = 1000)
    parser.add_argument('--bias', type = str, default = 'nobias')
    parser.add_argument('--beta', type = float, default = 0.1)

    parser.add_argument('--batchsize_trn', type = int, default = 10)
    parser.add_argument('--batchsize_val', type = int, default = 100)
    parser.add_argument('--batchsize_save', type = int, default = 10000)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'constant')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)

    parser.add_argument('--core_id', type = int, default = 0)
    args = parser.parse_args()
    print(f'running with {args}')

    # When using a pretrained model, make sure to specify a fixed max length
    if args.pretrained_model_name is not None:
        assert args.max_length is not None
    else:
        assert args.max_length is None
        assert args.model_type in ['gpt2','llama2']
        args.max_length = args.seq_len + 1 # add eos_token for GPT2 or LLaMa2

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Generate the dataset name and the run name
    if args.pretrained_model_name is not None:
        assert args.dataset_name is not None
        save_path = f'{args.base_dir}/attns/{args.pretrained_model_name}'
    else:
        args.dataset_name = gen_dataset_name(args)
        args.run_name = gen_run_name(args)
        save_path = f'{args.base_dir}/attns/{args.run_name}'
    os.makedirs(save_path+'/', exist_ok=True)

    # Fix the seed
    seed_everything(args.run_seed)

    # Load the tokenizer
    if args.pretrained_model_name is not None:
        args.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
    else:
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    args.tokenizer.pad_token = args.tokenizer.eos_token

    # Load the dataset
    dataset, data_loader = get_template_loaders(args)

    # Load the model
    if args.pretrained_model_name is not None:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
    model.to(args.device)
    model.eval()

    if args.graph_type=='tree-all':
        nlayers, nheads = args.num_layers, args.num_heads
    elif args.graph_type=='babylm':
        if model.config.model_type=='gpt2':
            nlayers, nheads = model.config.n_layer, model.config.n_head
        else:
            raise NotImplementedError
    num_sents = 0
    batch_id = 0
    attns = []
    for examples in data_loader:
        loaded_examples = examples.to(args.device)
        with torch.no_grad():
            outputs = model(input_ids=loaded_examples['input_ids'],
                            labels=loaded_examples['labels'],
                            attention_mask=loaded_examples['attention_mask'],
                            output_attentions=True)
        batch_size = len(loaded_examples['input_ids'])
        attns.append(torch.stack(outputs.attentions).transpose(0,1).cpu().numpy())
        num_sents += batch_size
        if num_sents >= args.batchsize_save:
            attns = np.concatenate(attns, axis=0)
            np.save(f'{save_path}/attns_{batch_id}.npy', attns)
            print(f'Saved batch {batch_id}')
            attns = []
            num_sents = 0
            batch_id += 1
    if len(attns)>0:
        attns = np.concatenate(attns, axis=0)
        np.save(f'{save_path}/attns_{batch_id}.npy', attns)
        print(f'Saved batch {batch_id}')