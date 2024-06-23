import numpy as np
import os
import argparse

import torch
from tokenizers import models, pre_tokenizers, normalizers, trainers, processors, Tokenizer

from utils import gen_dataset_name

def load_dataset(args):
    dirname = gen_dataset_name(args)
    trn_path = f'{args.base_dir}/dataset/{dirname}/trn.txt'
    val_path = f'{args.base_dir}/dataset/{dirname}/val.txt'
    tst_path = f'{args.base_dir}/dataset/{dirname}/tst.txt'
    return [trn_path, val_path, tst_path]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, required = True)
    parser.add_argument('--graph_type', type = str, required = True)
    parser.add_argument('--vocab_size', type = int, default = 5)
    parser.add_argument('--max_prob', type=float, default = 0.8)
    parser.add_argument('--seq_len', type = int, default = 16)
    parser.add_argument('--seed', type = int, default = 1234)
    args = parser.parse_args()
    print(f'running with {args}')
    args.base_dir = os.environ.get("MY_DATA_PATH")

    files = load_dataset(args)

    if args.model_type == 'bert':
        from transformers import BertTokenizerFast
        tokenizer_wrapper = BertTokenizerFast
        special_tokens = ['[UNK]','[CLS]','[SEP]','[PAD]','[MASK]']
        unk_token, bos_token, eos_token = '[UNK]', '[CLS]', '[SEP]'
    elif args.model_type == 'albert':
        from transformers import AlbertTokenizerFast
        tokenizer_wrapper = AlbertTokenizerFast
        special_tokens = ['<unk>','[CLS]','[SEP]','<pad>','[MASK]']
        unk_token, bos_token, eos_token = '<unk>', '[CLS]', '[SEP]'
    elif args.model_type=='gpt2':
        from transformers import GPT2TokenizerFast
        tokenizer_wrapper = GPT2TokenizerFast
        special_tokens = ['<|endoftext|>']
        unk_token = '<|endoftext|>'
    elif args.model_type=='llama2':
        from transformers import LlamaTokenizerFast
        tokenizer_wrapper = LlamaTokenizerFast
        special_tokens = ['<unk>','<s>','</s>']
        unk_token, bos_token, eos_token = '<unk>','<s>','</s>'
    else:
        raise NotImplementedError

    tokenizer = Tokenizer(models.WordLevel(unk_token=unk_token))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(),
                                                normalizers.Lowercase(),
                                                normalizers.StripAccents()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.WordLevelTrainer(vocab_size=128, special_tokens=special_tokens)
    tokenizer.train(files=files, trainer=trainer)

    if 'bert' in args.model_type:
        bos_token_id = tokenizer.token_to_id(bos_token)
        eos_token_id = tokenizer.token_to_id(eos_token)

        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{bos_token}:0 $A:0 {eos_token}:0",
            special_tokens=[(bos_token, bos_token_id), (eos_token, eos_token_id)]
            )

    wrapped_tokenizer = tokenizer_wrapper(tokenizer_object=tokenizer)
    wrapped_tokenizer.save_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_{gen_dataset_name(args)}')