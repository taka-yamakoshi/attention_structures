import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from copy import deepcopy
import json
import random

from utils_attn_loss import get_templates, calc_attn_loss_nback, calc_attn_loss_faiss
def evaluate(model, loaders, args, pretrained_model=None, pca_comps=None, index_list=None, xb_list=None):
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
                batch_size, num_heads, seq_len = outputs.attentions[0].shape[:-1]
                if args.bias=='direct':
                    outputs_pretrained = pretrained_model(input_ids=loaded_examples['input_ids'],
                                                          labels=loaded_examples['labels'],
                                                          attention_mask=loaded_examples['attention_mask'],
                                                          output_attentions=True)
                    attn_loss = torch.mean(torch.stack([torch.mean(torch.sum((attn1-attn2)**2,dim=(1,2,3)),dim=0)
                                                        for attn1, attn2 in zip(outputs.attentions, outputs_pretrained.attentions)]))
                else:
                    if args.graph_type.startswith('nback'):
                        if args.bias=='nobias':
                            # calculate the attention loss just for evaluation
                            args.bias = 'nback-all-1'
                            templates = get_templates(args, seq_len, batch_size, num_heads)
                            args.bias = 'nobias'
                            layer_ids = [1]
                        else:
                            templates = get_templates(args, seq_len, batch_size, num_heads)
                            layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                        attn_loss = calc_attn_loss_nback(outputs.attentions, templates, layer_ids)
                    elif args.graph_type.startswith('tree') or args.graph_type.startswith('babylm'):
                        if args.bias=='nobias':
                            attn_loss = torch.tensor([0]).to(args.device)
                        else:
                            layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                            attn_loss = calc_attn_loss_faiss(args, pca_comps, index_list, xb_list, outputs.attentions, layer_ids)
                main_loss_list.append(outputs.loss.item())
                attn_loss_list.append(attn_loss.item())
            main_out.append(np.mean(main_loss_list))
            attn_out.append(np.mean(attn_loss_list))
    return main_out, attn_out

def load_linzen(data_path):
    with open(f'{data_path}/subj_agr_filtered.text','r') as f:
        file = f.read().split('\n')[:-1]
    sents = [row[:-6] for row in file]

    with open(f'{data_path}/subj_agr_filtered.gold','r') as f:
        file = f.read().split('\n')[:-1]
    stats = [row.split('\t') for row in file]

    assert len(sents)==len(stats)
    head = ['sent_id','sent','verb_id','option_1','option_2','num_attr','prefix','sent_good','sent_bad']
    text = []
    for sent_id,(sent,stat) in enumerate(zip(sents,stats)):
        assert len(stat)==4
        verb_id = int(stat[0])
        option_1 = stat[1]
        option_2 = stat[2]
        num_attr = stat[3]
        prefix = ' '.join(sent.split(' ')[:verb_id])

        assert len(option_1.split(' '))==1 and len(option_2.split(' '))==1
        split_sent_good = sent.split(' ')
        assert split_sent_good[verb_id].lower()==option_1.lower(),f'"{split_sent_good}" does not match "{option_1}" in {sent_id}'
        split_sent_bad = deepcopy(split_sent_good)
        split_sent_bad[verb_id] = option_2
        sent_bad = ' '.join(split_sent_bad)
        text.append([sent_id,sent,verb_id,option_1,option_2,num_attr,prefix,sent,sent_bad])
    return head, text

def load_blimp(data_path,task):
    with open(f'{data_path}/{task}.jsonl','r') as f:
        file = f.readlines()
    head = ['pairID','sent_good','sent_bad']
    text = []
    for line in file:
        data = json.loads(line)
        sent_good = data['sentence_good']
        sent_bad = data['sentence_bad']
        pair_id = data['pairID']
        text.append([pair_id,sent_good,sent_bad])
    return head, text

def load_zorro(data_path, task):
    with open(f'{data_path}/{task}.txt','r') as f:
        file = f.readlines()
    assert len(file)==4000
    head = ['pairID','sent_good','sent_bad']
    text = []
    for pair_id in range(len(file)//2):
        sent_bad = file[2*pair_id]
        sent_good = file[2*pair_id+1]
        text.append([pair_id,sent_good,sent_bad])
    return head, text

def check_sent_length(tokenizer,sent,max_length):
    if max_length is None:
        return True
    else:
        input_ids = tokenizer(sent).input_ids
        return len(input_ids)<max_length

def eval_model_pairs(tokenizer,model,device,head,text,max_length=None):
    assert 'sent_good' in head and 'sent_bad' in head, 'header should contain sent_good and sent_bad'
    new_head = head + ['logprob_1', 'logprob_2']
    new_text = []
    num_sents = 0
    for line in text:
        sent_1, sent_2 = line[head.index('sent_good')],line[head.index('sent_bad')]
        if (not check_sent_length(tokenizer,sent_1,max_length)) or (not check_sent_length(tokenizer,sent_2,max_length)):
            continue
        logprob_1 = calc_prob_causal_lm(tokenizer,model,device,sent_1)
        logprob_2 = calc_prob_causal_lm(tokenizer,model,device,sent_2)
        new_text.append(line+[logprob_1,logprob_2])
        num_sents += 1
    print(f'{num_sents} sentences processed')
    df = pd.DataFrame(data=new_text,columns=new_head)
    df['acc'] = df['logprob_1']>df['logprob_2']
    return df

def eval_model_prfix(tokenizer,model,device,head,text,max_length=None):
    assert 'prefix' in head and 'option_1' in head and 'option_2' in head, 'header should contain prefix, option_1 and option_2'
    new_head = head + ['logprob_1', 'logprob_2']
    new_text = []
    num_sents = 0
    for line in text:
        prefix = line[head.index('prefix')]
        option_1, option_2 = line[head.index('option_1')], line[head.index('option_2')]
        if (not check_sent_length(tokenizer,prefix+' '+option_1,max_length)) or (not check_sent_length(tokenizer,prefix+' '+option_2,max_length)):
            continue
        logprob_1 = calc_prob_prefix(tokenizer,model,device,prefix,option_1)
        logprob_2 = calc_prob_prefix(tokenizer,model,device,prefix,option_2)
        new_text.append(line+[logprob_1,logprob_2])
        num_sents += 1
    print(f'{num_sents} sentences processed')
    df = pd.DataFrame(data=new_text,columns=new_head)
    df['acc'] = df['logprob_1']>df['logprob_2']
    return df

def calc_prob_prefix(tokenizer,model,device,prefix,cont):
    tokenized = tokenizer(prefix+' '+cont)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    start_id = len(tokenizer(prefix).input_ids)

    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(input_ids = torch.tensor(input_ids).unsqueeze(0).to(device),
                        attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(device))
    logprobs = F.log_softmax(outputs.logits.to('cpu'), dim = -1)
    assert logprobs.shape[0]==1 and logprobs.shape[1]==len(input_ids)

    cont_logprob = np.empty(len(input_ids)-start_id)
    for pos in range(start_id,len(input_ids)):
        cont_logprob[pos-start_id] = logprobs[0][pos-1][input_ids[pos]]
    return np.sum(cont_logprob).item()

def calc_prob_causal_lm(tokenizer,model,device,sent):
    tokenized = tokenizer(sent)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask

    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(input_ids = torch.tensor(input_ids).unsqueeze(0).to(device),
                        attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(device))
    logprobs = F.log_softmax(outputs.logits.to('cpu'), dim = -1)
    assert logprobs.shape[0]==1 and logprobs.shape[1]==len(input_ids)

    sent_logprob = np.empty(len(input_ids)-1)
    for pos, token in enumerate(input_ids):
        if pos==0:
            continue
        sent_logprob[pos-1] = logprobs[0][pos-1][token]
    return np.sum(sent_logprob).item()

def eval_model_linzen(data_path,tokenizer,model,device,num_samples=None,shuffle=True,max_length=None):
    head,text = load_linzen(data_path)
    if num_samples is not None:
        text = random.sample(text,num_samples)
    elif shuffle:
        text = random.sample(text,len(text))
    return eval_model_prfix(tokenizer,model,device,head,text,max_length)

def eval_model_blimp(data_path,task,tokenizer,model,device,num_samples=None,shuffle=True,max_length=None):
    head,text = load_blimp(data_path,task)
    if num_samples is not None:
        text = random.sample(text,num_samples)
    elif shuffle:
        text = random.sample(text,len(text))
    return eval_model_pairs(tokenizer,model,device,head,text,max_length)

def eval_model_zorro(data_path,task,tokenizer,model,device,num_samples=None,shuffle=True,max_length=None):
    head,text = load_zorro(data_path,task)
    if num_samples is not None:
        text = random.sample(text,num_samples)
    elif shuffle:
        text = random.sample(text,len(text))
    return eval_model_pairs(tokenizer,model,device,head,text,max_length)

def evaluate_linzen(model, args, num_samples=None):
    df = eval_model_linzen('../colorlessgreenRNNs/data/linzen_testset',
                           args.tokenizer,model,args.device,
                           num_samples=num_samples,max_length=args.max_length)
    df_group = df.filter(['num_attr','acc']).groupby(['num_attr'],as_index=False).mean()
    return {f'eval/linzen_test_{num_attr}':df_group.loc[lambda d: d['num_attr']==str(num_attr)]['acc'].item() for num_attr in range(5)}

def evaluate_linzen_agg(model, args, num_samples=None):
    df = eval_model_linzen('../colorlessgreenRNNs/data/linzen_testset',
                           args.tokenizer,model,args.device,
                           num_samples=num_samples,max_length=args.max_length)
    return {'eval/linzen_test':df['acc'].mean()}

def evaluate_blimp(model, args, tasks, num_samples=None):
    out = {}
    for task in tasks:
        df = eval_model_blimp('../blimp/data',task,
                              args.tokenizer,model,args.device,
                              num_samples=num_samples,max_length=args.max_length)
        out[f'eval/blimp_test_{task}'] = df['acc'].mean()
    return out

def evaluate_zorro(model, args, tasks, num_samples=None):
    out = {}
    for task in tasks:
        df = eval_model_zorro('../Zorro/sentences/babyberta',task,
                              args.tokenizer,model,args.device,
                              num_samples=num_samples,max_length=args.max_length)
        out[f'eval/zorro_test_{task}'] = df['acc'].mean()
    return out
