import numpy as np
import torch
import argparse
import os
import math

from transformers import AutoTokenizer, AutoModelForCausalLM,DataCollatorForLanguageModeling
from datasets import load_dataset

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
        config_kwargs['pad_token_id'] = args.tokenizer(args.tokenizer.pad_token).input_ids[1:-1][0]
        return BertConfig(**config_kwargs)
    elif model_type=='albert':
        from transformers import AlbertConfig
        assert len(args.tokenizer(args.tokenizer.pad_token).input_ids)==3
        assert len(args.tokenizer(args.tokenizer.bos_token).input_ids)==3
        assert len(args.tokenizer(args.tokenizer.eos_token).input_ids)==3
        config_kwargs['pad_token_id'] = args.tokenizer(args.tokenizer.pad_token).input_ids[1:-1][0]
        config_kwargs['bos_token_id'] = args.tokenizer(args.tokenizer.bos_token).input_ids[1:-1][0]
        config_kwargs['eos_token_id'] = args.tokenizer(args.tokenizer.eos_token).input_ids[1:-1][0]
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
    tokenized_dataset['trn'] = tokenized_dataset['trn'].filter(lambda example, idx: idx < args.datasize, with_indices=True)
    return tokenized_dataset

def gen_scheduler(optimizer, args):
    num_steps = args.num_epochs*math.ceil(args.datasize/args.batchsize_train)
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
    out = []
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            loss_list = []
            for examples in loader:
                loaded_examples = examples.to(args.device)
                outputs = model(input_ids=loaded_examples['input_ids'],
                                labels=loaded_examples['labels'],
                                attention_mask=loaded_examples['attention_mask'],
                                output_attentions=True)
                loss_list.append(outputs.loss.item()/len(loaded_examples['input_ids']))
            out.append(np.mean(loss_list))
    return out

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

    parser.add_argument('--batchsize_trn', type = int, default = 10)
    parser.add_argument('--batchsize_val', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'linear')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)

    parser.add_argument('--core_id', type = int, default = 0)
    args = parser.parse_args()
    print(f'running with {args}')
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.device = torch.device("cuda", index=int(args.core_id))

    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    args.run_name = gen_run_name(args)

    seed_everything(args.run_seed)

    import wandb
    wandb.init(project="attn_struct")
    config = wandb.config
    config.lr = args.lr
    config.datasize = args.datasize
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.hidden_size = args.hidden_size
    config.intermediate_size = args.intermediate_size
    config.batchsize_trn = args.batchsize_trn
    config.batchsize_val = args.batchsize_val

    args.dataset_name = gen_dataset_name(args)
    if args.graph_type.startswith('nback'):
        # Load the tokenizer for all n's
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')

    dataset = load_sentences(args)
    data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=False)

    trn_loader = torch.utils.data.DataLoader(dataset['trn'], batch_size=args.batchsize_trn,
                                             collate_fn=data_collator)
    val_loaders = []
    tst_loaders = []
    if args.graph_type.startswith('nback'):
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

    config = load_config(args.model_type,args)
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = gen_scheduler(optimizer, args)

    step_id = 0
    wandb.log(data={f'val-{n}':loss for n, loss in zip(np.arange(1,6), evaluate(model,val_loaders,args))},
              step=step_id)
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        model.train()
        dataset['trn'].shuffle(seed=args.run_seed+epoch)
        for examples in trn_loader:
            loaded_examples = examples.to(args.device)
            optimizer.zero_grad()
            outputs = model(input_ids=loaded_examples['input_ids'],
                            labels=loaded_examples['labels'],
                            attention_mask=loaded_examples['attention_mask'],
                            output_attentions=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_id += 1
            lr = scheduler.get_last_lr()[0]
            if step_id%100==0:
                wandb.log(data={
                        'train/lr':lr,
                        'train/loss':loss.item()},
                        step=step_id)

        val_loss = evaluate(model,val_loaders,args)
        wandb.log(data={f'val-{n}':loss for n, loss in zip(np.arange(1,6), val_loss)},step=step_id)
        model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")
        if np.mean(val_loss)<best_val_loss:
            best_val_loss = np.mean(val_loss)
            model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
            print(f'new best val loss: {best_val_loss}')

    wandb.log(data={f'tst-{n}':loss for n, loss in zip(np.arange(1,6), evaluate(model,tst_loaders,args))})