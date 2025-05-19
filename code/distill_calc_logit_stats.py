import numpy as np
import torch
import argparse
import os
import pandas as pd

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasize', type = int, default=None)
    args = parser.parse_args()
    print(f'running with {args}')

    args.version = 'emnlp'
    args.base_dir = os.environ.get("MY_DATA_PATH")
    os.makedirs(f'../distill_attn_stats/{args.version}/',exist_ok=True)

    kl_data = []
    head = ['bias','seed','taskID','sampleID','value']

    # Load teacher
    teacher_path = f"{args.base_dir}/distill_attns/{args.version}/gpt2"
    logprobs_teacher = torch.nn.functional.log_softmax(torch.tensor(np.load(f'{teacher_path}/logits.npy')),dim=-1).numpy()
    tokens_teacher = np.load(f'{teacher_path}/num_tokens.npy')
    ntasks, nsamples = tokens_teacher.shape
    assert len(logprobs_teacher.shape)==4
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
            logprobs_student = torch.nn.functional.log_softmax(torch.tensor(np.load(f'{student_path}/logits.npy')),dim=-1).numpy()
            tokens_student = np.load(f'{student_path}/num_tokens.npy')
            assert np.all(tokens_student==tokens_teacher)
            for task_id in range(ntasks):
                for sample_id in range(nsamples):
                    seq_len = tokens_student[task_id,sample_id]
                    logprob_s = logprobs_student[task_id,sample_id,:seq_len-1]
                    logprob_t = logprobs_teacher[task_id,sample_id,:seq_len-1]
                    kldiv = np.mean(np.sum(np.exp(logprob_t)*(-logprob_s),dim=-1))
                    kl_data.append([bias,seed,task_id,sample_id,kldiv])

    df = pd.DataFrame(kl_data,columns=head)
    df.to_csv(f'../distill_attn_stats/{args.version}/kl_{args.datasize}.csv')
