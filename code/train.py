import numpy as np
import torch
import argparse
import os
import math

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

import ot

from utils import gen_dataset_name, gen_run_name, seed_everything

def load_config(model_type,args):
    config_kwargs = {
                    'num_hidden_layers':args.num_layers,
                    'num_attention_heads':args.num_heads,
                    'hidden_size':args.hidden_size,
                    'intermediate_size':args.intermediate_size,
                    'vocab_size':args.tokenizer.vocab_size,
                    'max_position_embeddings':32,
                    'position_embedding_type':'absolute'
                    }
    if model_type=='bert':
        from transformers import BertConfig
        assert len(args.tokenizer(args.tokenizer.pad_token).input_ids)==3
        config_kwargs['pad_token_id'] = args.tokenizer(args.tokenizer.pad_token).input_ids[1]
        return BertConfig(**config_kwargs)
    elif model_type=='albert':
        from transformers import AlbertConfig
        assert len(args.tokenizer(args.tokenizer.pad_token).input_ids)==3
        assert len(args.tokenizer(args.tokenizer.bos_token).input_ids)==3
        assert len(args.tokenizer(args.tokenizer.eos_token).input_ids)==3
        config_kwargs['pad_token_id'] = args.tokenizer(args.tokenizer.pad_token).input_ids[1]
        config_kwargs['bos_token_id'] = args.tokenizer(args.tokenizer.bos_token).input_ids[1]
        config_kwargs['eos_token_id'] = args.tokenizer(args.tokenizer.eos_token).input_ids[1]
        config_kwargs['embedding_size'] = getattr(args,f'hidden_size')
        return AlbertConfig(**config_kwargs)
    elif model_type=='gpt2':
        from transformers import GPT2Config
        config_kwargs['n_embd'] = config_kwargs.pop('hidden_size')
        config_kwargs['n_inner'] = config_kwargs.pop('intermediate_size')
        config_kwargs['n_layer'] = config_kwargs.pop('num_hidden_layers')
        config_kwargs['n_head'] = config_kwargs.pop('num_attention_heads')
        config_kwargs['n_positions'] = config_kwargs.pop('max_position_embeddings')
        _ = config_kwargs.pop('position_embedding_type')
        assert len(args.tokenizer(args.tokenizer.bos_token).input_ids)==1
        assert len(args.tokenizer(args.tokenizer.eos_token).input_ids)==1
        config_kwargs['bos_token_id'] = args.tokenizer(args.tokenizer.bos_token).input_ids[0]
        config_kwargs['eos_token_id'] = args.tokenizer(args.tokenizer.eos_token).input_ids[0]
        return GPT2Config(**config_kwargs)
    elif model_type=='llama2':
        from transformers import LlamaConfig
        _ = config_kwargs.pop('position_embedding_type')
        assert len(args.tokenizer(args.tokenizer.bos_token).input_ids)==2
        assert len(args.tokenizer(args.tokenizer.eos_token).input_ids)==2
        assert args.tokenizer(args.tokenizer.bos_token).input_ids[0]==args.tokenizer(args.tokenizer.bos_token).input_ids[1]
        assert args.tokenizer(args.tokenizer.eos_token).input_ids[0]==args.tokenizer(args.tokenizer.bos_token).input_ids[1]
        config_kwargs['bos_token_id'] = args.tokenizer(args.tokenizer.bos_token).input_ids[1]
        config_kwargs['eos_token_id'] = args.tokenizer(args.tokenizer.eos_token).input_ids[1]
        return LlamaConfig(**config_kwargs)
    else:
        raise NotImplementedError

def tokenize_function(examples,tokenizer):
    tokens = tokenizer(examples["text"], return_special_tokens_mask=True)
    return {'input_ids':tokens.input_ids,
            'attention_mask':tokens.attention_mask}

def process_dataset(dataset,args,remove_cols):
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=10,
                                    remove_columns=remove_cols,
                                    fn_kwargs={'tokenizer':args.tokenizer})
    return tokenized_dataset

def load_sentences(args):
    data_files = {"trn": "trn.txt", "val": "val.txt", "tst": "tst.txt"}
    dataset = load_dataset(f'{args.base_dir}/dataset/{args.dataset_name}', data_files=data_files, cache_dir=args.cache_dir)
    remove_cols = ['text']

    tokenized_dataset = process_dataset(dataset, args, remove_cols)
    tokenized_dataset = tokenized_dataset.with_format("torch")
    tokenized_dataset['trn'] = tokenized_dataset['trn'].shuffle(seed=args.run_seed)
    tokenized_dataset['trn'] = tokenized_dataset['trn'].filter(lambda example, idx: idx < args.datasize, with_indices=True)
    return tokenized_dataset

def gen_scheduler(optimizer, args):
    num_steps = args.num_epochs*math.ceil(args.datasize/args.batchsize_trn)
    if args.scheduler_type=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    elif args.scheduler_type=='constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=num_steps)
    elif args.scheduler_type=='exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.scheduler_type=='cosine':
        T_max = max(num_steps//10,10000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_max)
    else:
        raise NotImplementedError
    return scheduler

def evaluate(model, loaders, args):
    main_out = []
    attn_out = []
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            main_loss_list = []
            attn_loss_list = []
            for examples in loader:
                loaded_examples = examples.to(args.device)
                outputs = model(input_ids=loaded_examples['input_ids'],
                                labels=loaded_examples['labels'],
                                attention_mask=loaded_examples['attention_mask'],
                                output_attentions=True)
                if args.bias=='nobias':
                    attn_loss = 0
                else:
                    templates = get_templates(args, outputs.attentions[0].shape[-1])
                    layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                    attn_loss = calc_attn_loss(outputs.attentions, templates, layer_ids)
                main_loss_list.append(outputs.loss.item())
                attn_loss_list.append(attn_loss)
            main_out.append(np.mean(main_loss_list))
            attn_out.append(np.mean(attn_loss_list))
    return main_out, attn_out

def get_template_nback(n:int, seq_len:int):
    # adjusted for next toke prediction
    mat = torch.tensor([[1 if j==i-(n-1) else 0
                         for j in range(seq_len)]
                         for i in range(seq_len)])
    return mat

def calc_wasserstein_distance(attn, temp):
    assert len(attn.shape)==2 and len(temp.shape)==2
    assert attn.shape[0]==attn.shape[1] and temp.shape[0]==temp.shape[1]
    seq_len = attn.shape[0]
    u_values = torch.arange(seq_len).repeat(seq_len,1).T
    v_values = torch.arange(seq_len).repeat(seq_len,1).T
    u_weights = attn.T
    v_weights = temp.T
    return torch.mean(ot.wasserstein_1d(u_values, v_values, u_weights, v_weights))

def get_templates(args, seq_len):
    ns = [1,2,3,4,5] if args.bias=='nback-all' else [int(args.bias.split('-')[1])]
    return [get_template_nback(n,seq_len).to(args.device) for n in ns]

def calc_attn_loss(attentions, templates, layer_ids):
    attn_loss = 0
    for layer_id, attn_layer in enumerate(attentions):
        if layer_id in layer_ids:
            for attn_batch in attn_layer:
                for attn in attn_batch:
                    for temp in templates:
                        attn_loss += calc_wasserstein_distance(attn, temp)/len(templates)
    return attn_loss

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True)
    parser.add_argument('--graph_type', type = str, required = True)
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

    # Initialize weights and biases with args
    import wandb
    wandb.require("core")
    wandb.init(project="attn_struct_nobias_2layers")
    wandb.config.update(args.__dict__)

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Generate the dataset name and the run name
    args.dataset_name = gen_dataset_name(args)
    args.run_name = gen_run_name(args)
    wandb.config.dataset_name = args.dataset_name
    wandb.config.run_name = args.run_name

    # Fix the seed
    seed_everything(args.run_seed)

    # Load the tokenizer
    if args.graph_type.startswith('nback'):
        # Load the tokenizer for all n's
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    if args.model_type in ['gpt2','llama2']:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    # Load the dataset and the data collator
    dataset = load_sentences(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=False)

    # Create the dataloaders
    val_loaders = []
    tst_loaders = []
    if args.graph_type.startswith('nback'):
        # Create separate dataloaders for all n's
        for n in range(1,6):
            args.dataset_name = f'nback-{n}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
            eval_dataset = load_sentences(args)
            val_loader = torch.utils.data.DataLoader(eval_dataset['val'], batch_size=args.batchsize_val,
                                                     collate_fn=data_collator)
            tst_loader = torch.utils.data.DataLoader(eval_dataset['tst'], batch_size=args.batchsize_val,
                                                     collate_fn=data_collator)
            val_loaders.append(val_loader)
            tst_loaders.append(tst_loader)
        args.dataset_name = gen_dataset_name(args)

    # Load the model
    config = load_config(args.model_type,args)
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)

    # Create the optimizer and the scheduler
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = gen_scheduler(optimizer, args)

    step_id = 0
    val_main_loss, val_attn_loss = evaluate(model,val_loaders,args)
    wandb.log(data={f'validation/val-main-{n}':loss
                    for n, loss in zip(np.arange(1,6), val_main_loss)},step=step_id)
    wandb.log(data={f'validation/val-attn-{n}':loss
                    for n, loss in zip(np.arange(1,6), val_attn_loss)},step=step_id)
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        model.train()
        dataset['trn'] = dataset['trn'].shuffle(seed=args.run_seed+epoch)
        trn_loader = torch.utils.data.DataLoader(dataset['trn'], batch_size=args.batchsize_trn,
                                                 collate_fn=data_collator)
        for examples in trn_loader:
            loaded_examples = examples.to(args.device)
            optimizer.zero_grad()
            outputs = model(input_ids=loaded_examples['input_ids'],
                            labels=loaded_examples['labels'],
                            attention_mask=loaded_examples['attention_mask'],
                            output_attentions=True)
            if args.bias=='nobias':
                attn_loss = 0
            else:
                templates = get_templates(args, outputs.attentions[0].shape[-1])
                layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                attn_loss = calc_attn_loss(outputs.attentions, templates, layer_ids)
            main_loss = outputs.loss
            loss = main_loss + args.beta*attn_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_id += 1
            lr = scheduler.get_last_lr()[0]
            if step_id%100==0:
                wandb.log(data={
                        'train/lr':lr,
                        'train/main_loss':main_loss.item(),
                        'train/attn_loss':attn_loss,
                        'train/loss':loss.item(),
                        },
                        step=step_id)
        if epoch%(max(args.num_epochs//10,1))==0:
            val_main_loss, val_attn_loss = evaluate(model,val_loaders,args)
            wandb.log(data={f'validation/val-main-{n}':loss 
                            for n, loss in zip(np.arange(1,6), val_main_loss)},step=step_id)
            wandb.log(data={f'validation/val-attn-{n}':loss 
                            for n, loss in zip(np.arange(1,6), val_attn_loss)},step=step_id)
            model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")
            val_loss = np.mean(val_main_loss)+args.beta*np.mean(val_attn_loss)
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
                print(f'new best val loss: {best_val_loss}')
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/last")
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args)
    wandb.log(data={f'tst-main-{n}':loss for n, loss in zip(np.arange(1,6), tst_main_loss)})
    wandb.log(data={f'tst-attn-{n}':loss for n, loss in zip(np.arange(1,6), tst_attn_loss)})