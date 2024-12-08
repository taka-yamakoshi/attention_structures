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
    if args.graph_type.startswith('nback') or args.graph_type.startswith('tree'):
        return f'{args.graph_type}_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}'
    else:
        assert args.graph_type=='babylm'
        return f'{args.graph_type}_{args.max_length}'

def gen_run_name(args):
    model_stat = f'{args.num_layers}_{args.num_heads}_{args.hidden_size}_{args.intermediate_size}'
    if args.pretrained_model_name is not None:
        bias_stat = f'{args.bias}_{args.pretrained_model_name}_{args.beta}'
    else:
        bias_stat = f'{args.bias}_{args.beta}'
    if args.bias not in ['nobias','direct']:
        bias_stat += f'_faiss_{args.faiss_index_name}'
    run_stat = f'{bias_stat}_{args.datasize}_{args.batchsize_trn}_{args.batchsize_val}_{args.lr}_{args.scheduler_type}_{args.num_epochs}_{args.run_seed}'
    return f'{args.model_type}_{gen_dataset_name(args)}_{model_stat}_{run_stat}'

def gen_run_name_key_amp(args):
    model_stat = f'{args.num_layers}_{args.num_heads}_{args.hidden_size}_{args.intermediate_size}'
    run_stat = f'key_amp_{args.layer_id}_{args.alpha}_{args.datasize}_{args.batchsize_trn}_{args.batchsize_val}_{args.lr}_{args.scheduler_type}_{args.num_epochs}_{args.run_seed}'
    return f'{args.model_type}_{gen_dataset_name(args)}_{model_stat}_{run_stat}'

def load_config(model_type,args):
    config_kwargs = {
                    'num_hidden_layers':args.num_layers,
                    'num_attention_heads':args.num_heads,
                    'hidden_size':args.hidden_size,
                    'intermediate_size':args.intermediate_size,
                    'vocab_size':args.tokenizer.vocab_size,
                    'max_position_embeddings':args.max_length,
                    'position_embedding_type':'absolute',
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