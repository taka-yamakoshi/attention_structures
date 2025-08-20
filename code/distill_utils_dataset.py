import torch
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

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
    data_files = {"trn": "trn.txt", "val": "val.txt", "tst": "tst.txt"}
    dataset = load_dataset(f'{args.base_dir}/babylm/{args.dataset_name}', data_files=data_files, cache_dir=args.cache_dir)
    dataset = dataset.filter(lambda example: len(example['text'].split(' '))>1) # make sure each sentence is at least two words or more.
    remove_cols = ['text']

    tokenized_dataset = process_dataset(dataset, args, remove_cols)
    tokenized_dataset = tokenized_dataset.with_format("torch")
    return tokenized_dataset

def get_data_loaders(args):
    # Load the dataset and the data collator
    dataset = load_sentences(args)

    dataset['trn'] = dataset['trn'].shuffle(seed=args.run_seed)
    dataset['val'] = dataset['val'].shuffle(seed=args.run_seed)
    dataset['tst'] = dataset['tst'].shuffle(seed=args.run_seed)

    dataset['trn'] = dataset['trn'].filter(lambda example, idx: idx < args.datasize, with_indices=True)
    dataset['val'] = dataset['val'].filter(lambda example, idx: idx < 10000, with_indices=True)
    dataset['tst'] = dataset['tst'].filter(lambda example, idx: idx < 10000, with_indices=True)
    if "bert" in args.model_type:
        data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=True,mlm_probability=0.40,mask_replace_prob=1.0,random_replace_prob=0.0)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=args.tokenizer,mlm=False)

    # Create the dataloaders
    val_loaders = []
    tst_loaders = []

    # Create separate dataloaders for all trees
    val_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=args.batchsize_val,
                                                collate_fn=data_collator)
    tst_loader = torch.utils.data.DataLoader(dataset['tst'], batch_size=args.batchsize_val,
                                                collate_fn=data_collator)
    val_loaders.append(val_loader)
    tst_loaders.append(tst_loader)

    return dataset, data_collator, val_loaders, tst_loaders
