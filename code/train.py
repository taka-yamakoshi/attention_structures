import numpy as np
import torch
import argparse
import os
import math
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import gen_dataset_name, gen_run_name, seed_everything, load_config
from utils_dataset import get_data_loaders
from utils_attn_loss import calc_faiss_index, get_templates, calc_attn_loss_nback, calc_attn_loss_faiss, calc_attn_loss_lsldg, load_lsldg, load_pca_comp, load_attns_meanstdv, calc_attn_loss_globalmean
from utils_eval import evaluate, evaluate_linzen, evaluate_blimp, evaluate_zorro

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
    parser.add_argument('--attn_loss_type', type = str, default = 'faiss', choices=['faiss','lsldg'])
    parser.add_argument('--beta', type = float, default = 0.1)
    #parser.add_argument('--nlist', type = int, default = 4096, choices=[4096,8192],
    #                    help="number of clusters/cells, see https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#if-below-1m-vectors-ivfk")
    #parser.add_argument('--nprobe', type = int, default = 16, help="number of cluseters/cells to visit to find the neighbors")
    parser.add_argument('--nneighbors', type = int, default = 64, help="number of neighbors used to calcualte the loss")
    parser.add_argument('--num_bases', type = int, default = 100, help="number of bases for lsldg")
    parser.add_argument('--knn', type = int, default = 10, help="k used to estimate the KL divergence")
    parser.add_argument('--sigma', type = float, default= 1.0, help="sigma for kernel density estimation")

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
    elif args.version=='klestimate':
        args.version += f'_{args.knn}'
    elif args.version=='kde':
        args.version += f'_{args.sigma:.2f}'

    #blimp_tasks = ['distractor_agreement_relational_noun','distractor_agreement_relative_clause']
    zorro_tasks = ['agreement_determiner_noun-across_1_adjective','agreement_determiner_noun-between_neighbors',
                   'filler-gap-wh_question_object','filler-gap-wh_question_subject']

    # When using a pretrained model, make sure to specify a fixed max length
    if args.graph_type in ['nback','tree']:
        if args.pretrained_model_name is not None:
            assert args.max_length is not None
        else:
            assert args.model_type in ['gpt2','llama2']
            args.max_length = args.seq_len + 1 # add eos_token for GPT2 or LLaMa2
    else:
        assert args.max_length is not None

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

    # Create faiss index name
    args.faiss_index_name, args.lsldg_name = None, None
    if args.bias not in ['nobias','direct']:
        if args.graph_type.startswith('tree') or args.graph_type.startswith('babylm'):
            if args.attn_loss_type == 'faiss':
                args.faiss_index_name = f"HNSW-{args.nneighbors}"
            elif args.attn_loss_type == 'lsldg':
                args.lsldg_name = f"lsldg-{args.num_bases}"

    # Generate the dataset name and the run name
    if args.dataset_name is None:
        args.dataset_name = gen_dataset_name(args)
    args.run_name = gen_run_name(args)
    wandb.config.dataset_name = args.dataset_name
    wandb.config.run_name = args.run_name

    # Fix the seed
    seed_everything(args.run_seed)
    args.rng = np.random.default_rng(args.run_seed)

    index_list, xb_list = None, None
    attns_mean, attns_stdv = None, None
    if args.bias.startswith('globalmean'):
        # Load average and standard deviation of attentoin matrices for the globalmean option
        attns_mean, attns_stdv = load_attns_meanstdv(args)
    elif args.bias not in ['nobias','direct']:
        if args.graph_type.startswith('tree') or args.graph_type.startswith('babylm'):
            # Load and train the faiss index
            # This is the most RAM-intensive part
            pca_comps = []
            for layer_id in range(args.num_layers):
                pca_comps_layer = []
                for head_id in range(args.num_heads):
                    pca_comps_layer.append(load_pca_comp(args, layer_id, head_id))
                pca_comps.append(pca_comps_layer)

            if args.attn_loss_type == 'faiss':
                index_list, xb_list = calc_faiss_index(args)
            elif args.attn_loss_type == 'lsldg':
                centers, thetas, sigmas = [], [], []
                for layer_id in range(args.num_layers):
                    center, theta, sigma = load_lsldg(args, layer_id)
                    centers.append(center)
                    thetas.append(theta)
                    sigmas.append(sigma)

    # Load the tokenizer
    if args.graph_type.startswith('nback'):
        # Load the tokenizer for all n's
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    elif args.graph_type.startswith('tree'):
        # Load the tokenizer for all trees
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    else:
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
    pretrained_model = None
    if args.bias=='direct':
        if args.graph_type.startswith('nback') or args.graph_type.startswith('tree') or args.pretrained_model_name.startswith('llama2'):
            pretrained_model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/models/{args.pretrained_model_name}/best')
        else:
            pretrained_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
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
    val_main_loss, val_attn_loss = evaluate(model,val_loaders,args,
                                            pretrained_model=pretrained_model,
                                            pca_comps=pca_comps,
                                            index_list=index_list,xb_list=xb_list,
                                            attns_mean=attns_mean,attns_stdv=attns_stdv,
                                            )
    wandb.log(data={f'validation/val-main-{i+1}':loss
                    for i, loss in enumerate(val_main_loss)},step=step_id)
    wandb.log(data={f'validation/val-attn-{i+1}':loss
                    for i, loss in enumerate(val_attn_loss)},step=step_id)
    if args.graph_type.startswith('babylm'):
        out_linzen = evaluate_linzen(model, args, num_samples=1000)
        #out_blimp = evaluate_blimp(model, args, blimp_tasks, num_samples=1000)
        out_zorro = evaluate_zorro(model, args, zorro_tasks, num_samples=1000)
        wandb.log(data=out_linzen, step=step_id)
        #wandb.log(data=out_blimp, step=step_id)
        wandb.log(data=out_zorro, step=step_id)
    model_out_dir = f"{args.base_dir}/models/{args.version}/{args.run_name}"
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
            batch_size, num_heads, seq_len = outputs.attentions[0].shape[:-1]
            if args.bias=='nobias':
                attn_loss = torch.tensor([0]).to(args.device)
            elif args.bias=='direct':
                with torch.no_grad():
                    outputs_pretrained = pretrained_model(input_ids=loaded_examples['input_ids'],
                                                          labels=loaded_examples['labels'],
                                                          attention_mask=loaded_examples['attention_mask'],
                                                          output_attentions=True)
                attn_loss = torch.mean(torch.stack([torch.mean(torch.sum((attn1-attn2)**2,dim=(1,2,3)),dim=0)
                                                    for attn1, attn2 in zip(outputs.attentions, outputs_pretrained.attentions)]))
            elif args.bias.startswith('globalmean'):
                attn_loss = calc_attn_loss_globalmean(args, outputs.attentions, attns_mean, attns_stdv)
            else:
                layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(val) for val in args.bias.split('-')[2:]]
                if args.graph_type.startswith('nback'):
                    templates = get_templates(args, seq_len, batch_size, num_heads)
                    attn_loss = calc_attn_loss_nback(outputs.attentions, templates, layer_ids)
                elif args.graph_type.startswith('tree') or args.graph_type.startswith('babylm'):
                    if args.attn_loss_type == 'faiss':
                        attn_loss = calc_attn_loss_faiss(args, pca_comps, index_list, xb_list, outputs.attentions, layer_ids)
                    elif args.attn_loss_type == 'lsldg':
                        attn_loss = calc_attn_loss_lsldg(args, pca_comps, centers, thetas, sigmas, outputs.attentions, layer_ids)

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
            val_main_loss, val_attn_loss = evaluate(model,val_loaders,args,
                                                    pretrained_model=pretrained_model,
                                                    pca_comps=pca_comps,
                                                    index_list=index_list,xb_list=xb_list,
                                                    attns_mean=attns_mean, attns_stdv=attns_stdv,
                                                    )
            wandb.log(data={f'validation/val-main-{i+1}':loss
                            for i, loss in enumerate(val_main_loss)},step=step_id)
            wandb.log(data={f'validation/val-attn-{i+1}':loss
                            for i, loss in enumerate(val_attn_loss)},step=step_id)
            if args.graph_type.startswith('babylm'):
                out_linzen = evaluate_linzen(model, args, num_samples=1000)
                #out_blimp = evaluate_blimp(model, args, blimp_tasks, num_samples=1000)
                out_zorro = evaluate_zorro(model, args, zorro_tasks, num_samples=1000)
                wandb.log(data=out_linzen, step=step_id)
                #wandb.log(data=out_blimp, step=step_id)
                wandb.log(data=out_zorro, step=step_id)
            model.save_pretrained(f"{model_out_dir}/ckpt-{step_id}")
            val_loss = np.mean(val_main_loss)+args.beta*np.mean(val_attn_loss)
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{model_out_dir}/best")
                print(f'new best val loss: {best_val_loss}')
    model.save_pretrained(f"{model_out_dir}/last")
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,
                                            pretrained_model=pretrained_model,
                                            pca_comps=pca_comps,
                                            index_list=index_list,xb_list=xb_list,
                                            attns_mean=attns_mean, attns_stdv=attns_stdv,
                                            )
    wandb.log(data={f'test-last/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-last/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})
    if args.graph_type.startswith('babylm'):
        out_linzen = evaluate_linzen(model, args)
        #out_blimp = evaluate_blimp(model, args, blimp_tasks)
        out_zorro = evaluate_zorro(model, args, zorro_tasks)
        wandb.log(data=out_linzen, step=step_id)
        #wandb.log(data=out_blimp, step=step_id)
        wandb.log(data=out_zorro, step=step_id)

    model = AutoModelForCausalLM.from_pretrained(f"{model_out_dir}/best")
    model.to(args.device)
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,
                                            pretrained_model=pretrained_model,
                                            pca_comps=pca_comps,
                                            index_list=index_list,xb_list=xb_list,
                                            attns_mean=attns_mean, attns_stdv=attns_stdv,
                                            )
    wandb.log(data={f'test-best/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-best/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})
    if args.graph_type.startswith('babylm'):
        out_linzen = evaluate_linzen(model, args)
        #out_blimp = evaluate_blimp(model, args, blimp_tasks)
        out_zorro = evaluate_zorro(model, args, zorro_tasks)
        wandb.log(data=out_linzen, step=step_id)
        #wandb.log(data=out_blimp, step=step_id)
        wandb.log(data=out_zorro, step=step_id)
