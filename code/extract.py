import numpy as np
import torch
import argparse
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

from utils import gen_dataset_name, gen_run_name, seed_everything
from train import load_sentences

def get_tree_template_loaders(args):
    # Load the dataset and the data collator
    dataset = load_sentences(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=False)
    data_loader = torch.utils.data.DataLoader(dataset['temps'], batch_size=args.batchsize_val,collate_fn=data_collator)
    return dataset, data_loader

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True)
    parser.add_argument('--graph_type', type = str, choices = ['tree-all'])
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
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'constant')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)

    parser.add_argument('--core_id', type = int, default = 0)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Generate the dataset name and the run name
    args.dataset_name = gen_dataset_name(args)
    args.run_name = gen_run_name(args)
    os.makedirs(f'{args.base_dir}/attns/{args.run_name}/', exist_ok=True)

    # Fix the seed
    seed_everything(args.run_seed)

    # Load the tokenizer
    args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    if args.model_type in ['gpt2','llama2']:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    dataset, data_loader = get_tree_template_loaders(args)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
    model.to(args.device)
    model.eval()

    assert args.model_type in ['gpt2','llama2'] and args.graph_type=='tree-all'
    attns = torch.zeros((len(dataset["temps"]),args.num_layers,args.num_heads,args.seq_len+1,args.seq_len+1))
    num_sents = 0
    for examples in data_loader:
        loaded_examples = examples.to(args.device)
        outputs = model(input_ids=loaded_examples['input_ids'],
                        labels=loaded_examples['labels'],
                        attention_mask=loaded_examples['attention_mask'],
                        output_attentions=True)
        batch_size = len(loaded_examples['input_ids'])
        for layer_id in range(args.num_layers):
            attns[num_sents:num_sents+batch_size,layer_id,:,:,:] = outputs.attentions[layer_id]
        num_sents += batch_size
    np.save(f'{args.base_dir}/attns/{args.run_name}/attns.npy', attns.cpu().numpy())