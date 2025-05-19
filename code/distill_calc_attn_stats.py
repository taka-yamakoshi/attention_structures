import numpy as np
import torch
import argparse
import os
import pandas as pd

def calc_entropy(attn):
    return -np.sum(attn*np.log(attn+1e-10),axis=-1).mean(axis=-1)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasize', type = int, default=None)
    args = parser.parse_args()
    print(f'running with {args}')

    args.version = 'emnlp'
    args.base_dir = os.environ.get("MY_DATA_PATH")
    os.makedirs(f'../distill_attn_stats/{args.version}/',exist_ok=True)

    ent_data = []
    ent_head = ['modelType','bias','seed','taskID','sampleID','layerID','headID','value']
    
    attn_loss_data =[]
    attn_loss_head = ['bias','seed','taskID','sampleID','layerID','headID','value']

    ave_attn_loss_data = []
    ave_attn_loss_head = ['bias','seed','taskID','sampleID','value']

    # Load teacher
    teacher_path = f"{args.base_dir}/distill_attns/{args.version}/gpt2"
    attns_teacher = np.load(f'{teacher_path}/attns.npy')
    tokens_teacher = np.load(f'{teacher_path}/num_tokens.npy')
    ntasks, nsamples = tokens_teacher.shape
    for task_id in range(ntasks):
        for sample_id in range(nsamples):
                seq_len = tokens_teacher[task_id,sample_id]
                attn_t = attns_teacher[task_id,sample_id,:,:,:seq_len-1,:seq_len-1]
                ent_t = calc_entropy(attn_t)
                for layer_id in range(12):
                    for head_id in range(12):
                        ent_data.append(['teacher', 'attns_0.0', 10, task_id, sample_id, layer_id, head_id, ent_t[layer_id,head_id]])
    print('Finished loading the teacher')

    # Load students
    if args.datasize==100000:
        dataset_name, nepochs = 'babylm_10M', 50
    elif args.datasize==500000:
        dataset_name, nepochs = 'babylm_10M', 10
    elif args.datasize==1000000:
        dataset_name, nepochs = 'babylm_100M', 5
    else:
        raise NotImplementedError

    for bias in ['attns_0.0','attns_1.0','logits_10.0']:
        for seed in [1000,2000,3000]:
            print(f'seed {seed} of {bias}')
            model_name = f'gpt2_{dataset_name}-128_12-12-768-3072_gpt2_{bias}_{args.datasize}-32-100_0.0002-linear-{nepochs}_{seed}'
            student_path = f"{args.base_dir}/distill_attns/{args.version}/{model_name}"
            attns_student = np.load(f'{student_path}/attns.npy')
            tokens_student = np.load(f'{student_path}/num_tokens.npy')
            assert np.all(tokens_student==tokens_teacher)
            for task_id in range(ntasks):
                for sample_id in range(nsamples):
                    seq_len = tokens_student[task_id,sample_id]
                    attn_s = attns_student[task_id,sample_id,:,:,:seq_len-1,:seq_len-1]
                    attn_t = attns_teacher[task_id,sample_id,:,:,:seq_len-1,:seq_len-1]

                    ent_s = calc_entropy(attn_s)
                    attn_loss = np.sum((attn_s-attn_t)**2,axis=-1).mean(axis=-1)
                    ave_attn_loss = np.sum((attn_s.mean(axis=(0,1))-attn_t.mean(axis=(0,1)))**2,axis=-1).mean(axis=-1)
                    for layer_id in range(12):
                        for head_id in range(12):
                            ent_data.append(['student', bias, seed, task_id, sample_id, layer_id, head_id, ent_s[layer_id,head_id]])
                            attn_loss_data.append([bias, seed, task_id, sample_id, layer_id, head_id, attn_loss[layer_id,head_id]])
                    ave_attn_loss_data.append([bias, seed, task_id, sample_id, ave_attn_loss])

    df_ent = pd.DataFrame(ent_data,columns=ent_head)
    df_ent.to_csv(f'../distill_attn_stats/{args.version}/ent_{args.datasize}.csv')

    df_attn_loss = pd.DataFrame(attn_loss_data,columns=attn_loss_head)    
    df_attn_loss.to_csv(f'../distill_attn_stats/{args.version}/attn_loss_{args.datasize}.csv')

    df_ave_attn_loss = pd.DataFrame(ave_attn_loss_data,columns=ave_attn_loss_head)
    df_ave_attn_loss.to_csv(f'../distill_attn_stats/{args.version}/ave_attn_loss_{args.datasize}.csv')
