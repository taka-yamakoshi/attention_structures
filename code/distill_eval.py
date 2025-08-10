import numpy as np
import torch
import argparse
import os
import glob
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from distill_utils import seed_everything
from distill_utils_eval import evaluate_linzen_agg, evaluate_blimp, evaluate_zorro

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type = str, default=None)
    parser.add_argument('--model_name', type = str, default=None)
    parser.add_argument('--max_length', type = int, default=None)
    parser.add_argument('--run_seed', type = int, default=1234)
    parser.add_argument('--core_id', type = int, default = 0)
    parser.add_argument('--version', type = str, required=True)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(f'../distill_results/{args.version}', exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Fix the seed
    seed_everything(args.run_seed)
    args.rng = np.random.default_rng(args.run_seed)

    # Load the tokenizer
    if args.model_type=='gpt2':
        args.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
    elif args.model_type=='llama2':
        args.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                        cache_dir=args.cache_dir, token=os.environ.get('HF_TOKEN'))
    elif args.model_type=='bert':
        args.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir=args.cache_dir)
    elif args.model_type=='roberta':
        args.tokenizer = AutoTokenizer.from_pretrained('roberta-base', cache_dir=args.cache_dir)
    else:
        raise NotImplementedError
    args.tokenizer.pad_token = args.tokenizer.eos_token

    if args.model_name in ['gpt2','bert-base-uncased','roberta-base']:
        model = AutoModelForCausalLM.from_pretrained(f'{args.model_name}',cache_dir=args.cache_dir,token=os.environ.get('HF_TOKEN'))
    else:
        model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/distill_models/{args.version}/{args.model_name}/best')
    model.eval()
    model.to(args.device)

    blimp_prefix = '../blimp/data'
    zorro_prefix = '../Zorro/sentences/babyberta'
    blimp_tasks = [file.replace(blimp_prefix+'/','').replace('.jsonl','') for file in glob.glob(f'{blimp_prefix}/*.jsonl')]
    zorro_tasks = [file.replace(zorro_prefix+'/','').replace('.txt','') for file in glob.glob(f'{zorro_prefix}/*.txt')]
    if 'bert' in args.model_type:
        blimp_tasks_to_remove = [
            "principle_A_domain_3",
            "principle_A_reconstruction",
            "ellipsis_n_bar_1",
            "ellipsis_n_bar_2",
            "wh_questions_object_gap",
            "wh_questions_subject_gap",
            "wh_questions_subject_gap_long_distance",
            "adjunct_island",
            "complex_NP_island",
            "coordinate_structure_constraint_complex_left_branch",
            "coordinate_structure_constraint_object_extraction",
            "left_branch_island_echo_question",
            "left_branch_island_simple_question",
            "sentential_subject_island",
            "matrix_question_npi_licensor_present",
            "only_npi_scope",
            "sentential_negation_npi_scope",
            "existential_there_quantifiers_2",
            ]
        zorro_tasks_to_remove = [
            "argument_structure-dropped_argument",
            "argument_structure-swapped_arguments",
            "case-subjective_pronoun",
            "ellipsis-n_bar",
            "filler-gap-wh_question_object",
            "filler-gap-wh_question_subject",
            "island-effects-adjunct_island",
            "island-effects-coordinate_structure_constraint",
            "npi_licensing-matrix_question",
            ]
        blimp_tasks = [task for task in blimp_tasks if task not in blimp_tasks_to_remove]
        zorro_tasks = [task for task in zorro_tasks if task not in zorro_tasks_to_remove]
        mask_eval = True
    else:
        mask_eval = False
    out_linzen = evaluate_linzen_agg(model, args, mask_eval=mask_eval)
    out_blimp = evaluate_blimp(model, args, blimp_tasks, mask_eval=mask_eval)
    out_zorro = evaluate_zorro(model, args, zorro_tasks, mask_eval=mask_eval)

    tasks = []
    perfs = []
    for key, val in out_linzen.items():
        tasks.append(key)
        perfs.append(val)
    for key, val in out_blimp.items():
        tasks.append(key)
        perfs.append(val)
    for key, val in out_zorro.items():
        tasks.append(key)
        perfs.append(val)
    df = pd.DataFrame.from_dict({'tasks': tasks, 'acc': perfs})
    df.to_csv(f'../distill_results/{args.version}/{args.model_name}.csv', index=False)