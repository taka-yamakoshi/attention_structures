import numpy as np
import torch
import argparse
import os
import math
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from distill_utils_attn_loss import shuffle_attns_all
from distill_utils import gen_run_name, seed_everything, load_config
from distill_utils_dataset import get_data_loaders
from distill_utils_eval import evaluate, evaluate_linzen, evaluate_zorro

def gen_scheduler(optimizer, args):
    if args.scheduler_type=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.num_steps)
    elif args.scheduler_type=='constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.num_steps)
    elif args.scheduler_type=='cosine':
        T_0 = args.num_steps//args.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0)
    else:
        raise NotImplementedError
    return scheduler

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type = str, default=None)
    parser.add_argument('--dataset_name', type = str, default=None)
    parser.add_argument('--max_length', type = int, default=None)

    parser.add_argument('--model_type', type = str, required = True)
    parser.add_argument('--num_layers', type = int, default = 1)
    parser.add_argument('--num_heads', type = int, default = 1)
    parser.add_argument('--hidden_size', type = int, default = 128)
    parser.add_argument('--intermediate_size', type = int, default = 512)

    parser.add_argument('--datasize', type = int, default = 1000)
    parser.add_argument('--shuffle', type = str)
    parser.add_argument('--beta', type = float, default = 0.1)

    parser.add_argument('--batchsize_trn', type = int, default = 10)
    parser.add_argument('--batchsize_val', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'constant')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)
    parser.add_argument('--wandb_name', type = str, default = 'attn_struct')

    parser.add_argument('--core_id', type = int, default = 0)
    parser.add_argument('--version', type = str, default = None)
    args = parser.parse_args()
    print(f'running with {args}')

    if args.version is None:
        import datetime
        args.version = datetime.date.today().strftime('%Y-%m-%d')

    zorro_tasks = ['agreement_determiner_noun-across_1_adjective','agreement_determiner_noun-between_neighbors',
                   'filler-gap-wh_question_object','filler-gap-wh_question_subject']

    # Initialize weights and biases with args
    import wandb
    wandb.login(key=os.environ.get("WANDB_KEY"))
    wandb.require("core")
    wandb.init(project=args.wandb_name)
    wandb.config.update(args.__dict__)

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Generate the dataset name and the run name
    args.dataset_name += f'_{args.max_length}'
    args.run_name = gen_run_name(args)
    wandb.config.dataset_name = args.dataset_name
    wandb.config.run_name = args.run_name

    # Fix the seed
    seed_everything(args.run_seed)
    args.rng = np.random.default_rng(args.run_seed)

    # Load the tokenizer
    if args.model_type=='gpt2':
        args.tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir=args.cache_dir)
    elif args.model_type=='llama2':
        args.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',
                                                        cache_dir=args.cache_dir, token=os.environ.get('HF_TOKEN'))
    else:
        raise NotImplementedError
    assert args.model_type in ['gpt2','llama2']
    args.tokenizer.pad_token = args.tokenizer.eos_token

    # Load the model if necessary
    if args.pretrained_model_name.startswith('teacher'):
        pretrained_model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/teachers/{args.pretrained_model_name}/best')
    else:
        pretrained_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, 
                                                                cache_dir=args.cache_dir, token=os.environ.get('HF_TOKEN'))
    pretrained_model.eval()
    pretrained_model.to(args.device)

    dataset, data_collator, val_loaders, tst_loaders = get_data_loaders(args)
    args.num_steps = args.num_epochs*math.ceil(len(dataset['trn'])/args.batchsize_trn)
    print(f'{args.num_steps} steps')

    # Load the model
    config = load_config(args.model_type,args)
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)

    # Create the optimizer and the scheduler
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = gen_scheduler(optimizer, args)

    step_id = 0
    val_main_loss, val_attn_loss = evaluate(model,val_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'validation/val-main-{i+1}':loss
                    for i, loss in enumerate(val_main_loss)},step=step_id)
    wandb.log(data={f'validation/val-attn-{i+1}':loss
                    for i, loss in enumerate(val_attn_loss)},step=step_id)
    if args.graph_type.startswith('babylm'):
        out_linzen = evaluate_linzen(model, args, num_samples=1000)
        out_zorro = evaluate_zorro(model, args, zorro_tasks, num_samples=1000)
        wandb.log(data=out_linzen, step=step_id)
        wandb.log(data=out_zorro, step=step_id)
    model_out_dir = f"{args.base_dir}/distill_models/{args.version}/{args.run_name}"
    model.save_pretrained(f"{model_out_dir}/ckpt-{step_id}")

    best_val_loss = np.inf
    print(f'Started training at {time.ctime()}')
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
            with torch.no_grad():
                outputs_pretrained = pretrained_model(input_ids=loaded_examples['input_ids'],
                                                      labels=loaded_examples['labels'],
                                                      attention_mask=loaded_examples['attention_mask'],
                                                      output_attentions=True)
            attns = torch.stack(outputs.attentions)
            pretrained_attns = torch.stack(outputs_pretrained.attentions)
            attn_loss = torch.mean(torch.sum((attns-shuffle_attns_all(pretrained_attns, args))**2,dim=(2,3,4)))

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
                        'train/attn_loss':attn_loss.item(),
                        'train/loss':loss.item(),
                        },
                        step=step_id)
        if epoch%(max(args.num_epochs//10,1))==0:
            val_main_loss, val_attn_loss = evaluate(model,val_loaders,args,pretrained_model=pretrained_model)
            wandb.log(data={f'validation/val-main-{i+1}':loss
                            for i, loss in enumerate(val_main_loss)},step=step_id)
            wandb.log(data={f'validation/val-attn-{i+1}':loss
                            for i, loss in enumerate(val_attn_loss)},step=step_id)
            out_linzen = evaluate_linzen(model, args, num_samples=1000)
            out_zorro = evaluate_zorro(model, args, zorro_tasks, num_samples=1000)
            wandb.log(data=out_linzen, step=step_id)
            wandb.log(data=out_zorro, step=step_id)
            model.save_pretrained(f"{model_out_dir}/ckpt-{step_id}")
            val_loss = np.mean(val_main_loss)+args.beta*np.mean(val_attn_loss)
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{model_out_dir}/best")
                print(f'new best val loss: {best_val_loss}')
    model.save_pretrained(f"{model_out_dir}/last")
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'test-last/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-last/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})
    out_linzen = evaluate_linzen(model, args)
    out_zorro = evaluate_zorro(model, args, zorro_tasks)
    wandb.log(data=out_linzen, step=step_id)
    wandb.log(data=out_zorro, step=step_id)

    model = AutoModelForCausalLM.from_pretrained(f"{model_out_dir}/best")
    model.to(args.device)
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'test-best/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-best/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})
    out_linzen = evaluate_linzen(model, args)
    out_zorro = evaluate_zorro(model, args, zorro_tasks)
    wandb.log(data=out_linzen, step=step_id)
    wandb.log(data=out_zorro, step=step_id)
