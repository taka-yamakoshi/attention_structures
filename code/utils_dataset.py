import torch
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

from utils import gen_dataset_name

def tokenize_function(examples,tokenizer,max_length):
    tokens = tokenizer(examples["text"],
                       padding='max_length', truncation=True, max_length=max_length,
                       return_special_tokens_mask=True)
    return {'input_ids':tokens.input_ids,
            'attention_mask':tokens.attention_mask}

def process_dataset(dataset,args,remove_cols):
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=10,
                                    remove_columns=remove_cols,
                                    fn_kwargs={'tokenizer':args.tokenizer, 'max_length':args.max_length})
    return tokenized_dataset

def load_sentences(args):
    if args.graph_type == 'tree-all':
        data_files = {"trn": "trn.txt", "val": "val.txt", "tst": "tst.txt", "ex_val": "ex_val.txt", "ex_tst":"ex_tst.txt", "temps": "templates.txt"}
    else:
        data_files = {"trn": "trn.txt", "val": "val.txt", "tst": "tst.txt"}

    if args.graph_type == 'babylm':
        dataset = load_dataset(f'{args.base_dir}/babylm/{args.dataset_name}', data_files=data_files, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(f'{args.base_dir}/dataset/{args.dataset_name}', data_files=data_files, cache_dir=args.cache_dir)
    remove_cols = ['text']

    tokenized_dataset = process_dataset(dataset, args, remove_cols)
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset

def get_data_loaders(args):
    # Load the dataset and the data collator
    dataset = load_sentences(args)
    dataset['trn'] = dataset['trn'].filter(lambda example, idx: idx < args.datasize, with_indices=True)
    dataset['val'] = dataset['val'].filter(lambda example, idx: idx < 10000, with_indices=True)
    dataset['tst'] = dataset['tst'].filter(lambda example, idx: idx < 10000, with_indices=True)
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

    elif args.graph_type.startswith('tree'):
        # Create separate dataloaders for all trees
        val_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=args.batchsize_val,
                                                 collate_fn=data_collator)
        tst_loader = torch.utils.data.DataLoader(dataset['tst'], batch_size=args.batchsize_val,
                                                 collate_fn=data_collator)
        val_loaders.append(val_loader)
        tst_loaders.append(tst_loader)
        if args.graph_type=='tree-all':
            ex_val_loader = torch.utils.data.DataLoader(dataset['ex_val'], batch_size=args.batchsize_val,
                                                        collate_fn=data_collator)
            ex_tst_loader = torch.utils.data.DataLoader(dataset['ex_tst'], batch_size=args.batchsize_val,
                                                        collate_fn=data_collator)
            val_loaders.append(ex_val_loader)
            tst_loaders.append(ex_tst_loader)

    elif args.graph_type=='babylm':
        # Create separate dataloaders for all trees
        val_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=args.batchsize_val,
                                                 collate_fn=data_collator)
        tst_loader = torch.utils.data.DataLoader(dataset['tst'], batch_size=args.batchsize_val,
                                                 collate_fn=data_collator)
        val_loaders.append(val_loader)
        tst_loaders.append(tst_loader)

    return dataset, data_collator, val_loaders, tst_loaders