import numpy as np
import torch
import argparse
import os
import math

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from datasets import load_dataset

#import ot

from utils import gen_dataset_name, gen_run_name, seed_everything, load_config

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
    #tokenized_dataset['trn'] = tokenized_dataset['trn'].shuffle(seed=args.run_seed)
    tokenized_dataset['trn'] = tokenized_dataset['trn'].filter(lambda example, idx: idx < args.datasize, with_indices=True)
    return tokenized_dataset

def get_data_loaders(args):
    # Load the dataset and the data collator
    dataset = load_sentences(args)
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

def gen_scheduler(optimizer, args):
    num_steps = args.num_epochs*math.ceil(args.datasize/args.batchsize_trn)
    if args.scheduler_type=='linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
    elif args.scheduler_type=='constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=num_steps)
    elif args.scheduler_type=='exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.scheduler_type=='cosine':
        T_max = max(num_steps//10,10000)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_max)
    else:
        raise NotImplementedError
    return scheduler

def evaluate(model, loaders, args, pretrained_model=None):
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
                    elif args.graph_type.startswith('tree'):
                        if args.bias=='nobias':
                            attn_loss = torch.tensor([0]).to(args.device)
                        else:
                            layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                            attn_loss = calc_attn_loss_faiss(args, index_list, xb_list, outputs.attentions, layer_ids)
                main_loss_list.append(outputs.loss.item())
                attn_loss_list.append(attn_loss.item())
            main_out.append(np.mean(main_loss_list))
            attn_out.append(np.mean(attn_loss_list))
    return main_out, attn_out

def get_template_nback(n:int, seq_len:int, batch_size:int, num_heads:int):
    # adjusted for next toke prediction
    mat = torch.tensor([[[[1 if j==i-(n-1) else 0
                           for j in range(seq_len)]
                           for i in range(seq_len)]
                           for _ in range(num_heads)]
                           for _ in range(batch_size)])
    return mat

def calc_wasserstein_distance(attns:torch.Tensor, temps:torch.Tensor):
    assert attns.shape == temps.shape
    assert len(attns.shape)  == 5
    seq_len = attns.shape[-1]
    u_values = torch.arange(seq_len).repeat(*attns.shape[:3],seq_len,1).permute((4,0,1,2,3))
    v_values = torch.arange(seq_len).repeat(*temps.shape[:3],seq_len,1).permute((4,0,1,2,3))
    u_weights = attns.permute((4,0,1,2,3))
    v_weights = temps.permute((4,0,1,2,3))
    device = u_weights.device
    return ot.wasserstein_1d(u_values.to(device), v_values.to(device), u_weights, v_weights)

def adjust_layer_assignment(attns, nlayers_new):
    assert len(attns.shape)==5
    batch_size, nlayers, nheads, seqlen, _ = attns.shape
    assert nlayers%nlayers_new==0
    ratio = nlayers//nlayers_new

    new_attns = np.empty((ratio*batch_size, nlayers_new, nheads, seqlen, seqlen))
    for layer_id in range(nlayers_new):
        new_attns[:,layer_id,:,:,:] = attns[:,ratio*layer_id:ratio*(layer_id+1),:,:,:].reshape((ratio*batch_size, nheads, seqlen, seqlen))
    del attns
    return new_attns

def load_attns(args):
    if args.graph_type.startswith('tree') or args.graph_type.startswith('nback'):
        attns_path = f'{args.base_dir}/attns/{args.attn_run_name}/attns.npy'
        attns = np.load(attns_path)
    else:
        import glob
        attns_path_list = glob.glob(f'{args.base_dir}/attns/{args.attn_run_name}/*.npy')
        attns = []
        for attns_path in attns_path_list:
            loaded_attns = np.load(attns_path)
            attns.append(loaded_attns)
        attns = np.concatenate(attns, axis=0)
    return attns

def calc_faiss_index(args):
    import faiss
    attns = load_attns(args)
    attns = adjust_layer_assignment(attns, args.num_layers)
    # attns.shape = (batch_size, nlayers, nheads, seqlen, seqlen)
    assert len(attns.shape)==5
    index_list = []
    xb_list = []
    for layer_id in range(args.num_layers):
        xb = attns[:,layer_id,:,:,:]
        xb = xb.reshape((xb.shape[0]*xb.shape[1],xb.shape[2]*xb.shape[3]))
        rand_ids = args.rng.permutation(xb.shape[0])
        xb = xb[rand_ids]
        xb = xb.astype('float32')
        _, d = xb.shape

        faiss_index = faiss.index_factory(d, "PCA256,HNSW32,Flat")
        faiss_index.train(xb)
        faiss_index.add(xb)
        index_list.append(faiss_index)
        xb_list.append(xb)
    return index_list, xb_list

def calc_attn_loss_faiss(args, index_list, xb_list, attns, layer_ids):
    # attns.shape = (nlayers, batchsize, nheads, seqlen, seqlen)
    attn_loss = 0
    for layer_id in layer_ids:
        attn = attns[layer_id]
        faiss_index = index_list[layer_id]
        xb = xb_list[layer_id]
        assert len(attn.shape)==4
        xq = attn.clone().detach().cpu().numpy().astype('float32')
        xq = xq.reshape((xq.shape[0]*xq.shape[1],xq.shape[2]*xq.shape[3]))
        _, I = faiss_index.search(xq, args.num_neighbors)

        # xn.shape = (batchsize*nheads, neighbors, seqlen*seqlen)
        xn = torch.tensor(np.array([[xb[sample_id] for sample_id in line] for line in I]),device=args.device)
        assert len(xn.shape)==3 and xn.shape[1]==args.num_neighbors

        # attn.shape = (batchsize*nheads, seqlen*seqlen)
        attn = attn.view(attn.shape[0]*attn.shape[1],attn.shape[2]*attn.shape[3])
        assert attn.shape[0]==xn.shape[0] and attn.shape[1]==xn.shape[2]

        # dist.shape = (batchsize*nheads, neighbors)
        dist = torch.sqrt(((attn.unsqueeze(1)-xn)**2).sum(dim=-1))

        min_dist = -torch.logsumexp(-dist, dim=1)
        attn_loss += torch.mean(min_dist)
    return attn_loss

def get_templates(args, seq_len, batch_size, num_heads):
    ns = [1,2,3,4,5] if args.bias.startswith('nback-all') else [int(args.bias.split('-')[1])]
    return torch.stack([get_template_nback(n,seq_len,batch_size,num_heads).to(args.device) for n in ns])

def calc_attn_loss_nback(attentions, temps, layer_ids):
    attn_loss = 0
    for layer_id, attn_layer in enumerate(attentions):
        if layer_id in layer_ids:
            assert len(attn_layer.shape) == 4 and attn_layer.shape[2] == attn_layer.shape[3]
            attns = torch.stack([attn_layer for _ in range(temps.shape[0])])

            dist = calc_wasserstein_distance(attns, temps)
            assert dist.shape == attns.shape[:-1]
            min_dist = -torch.logsumexp(-dist, dim=0)
            assert min_dist.shape == attns.shape[1:-1]
            attn_loss += torch.mean(min_dist)
    return attn_loss

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
    parser.add_argument('--beta', type = float, default = 0.1)
    parser.add_argument('--num_neighbors', type = int, default = 100)

    parser.add_argument('--batchsize_trn', type = int, default = 10)
    parser.add_argument('--batchsize_val', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'constant')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)
    parser.add_argument('--wandb_name', type = str, default = 'attn_struct')

    parser.add_argument('--core_id', type = int, default = 0)
    args = parser.parse_args()
    print(f'running with {args}')

    # When using a pretrained model, make sure to specify a fixed max length
    if args.pretrained_model_name is not None:
        assert args.max_length is not None
    else:
        assert args.max_length is None
        assert args.model_type in ['gpt2','llama2']
        args.max_length = args.seq_len + 1 # add eos_token for GPT2 or LLaMa2

    # Initialize weights and biases with args
    import wandb
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
    if args.dataset_name is None:
        args.dataset_name = gen_dataset_name(args)
    args.run_name = gen_run_name(args)
    wandb.config.dataset_name = args.dataset_name
    wandb.config.run_name = args.run_name

    # Fix the seed
    seed_everything(args.run_seed)
    args.rng = np.random.default_rng(args.run_seed)

    # Load the tokenizer
    if args.graph_type.startswith('nback'):
        # Load the tokenizer for all n's
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    elif args.graph_type.startswith('tree'):
        # Load the tokenizer for all trees
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    else:
        args.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
    assert args.model_type in ['gpt2','llama2']
    args.tokenizer.pad_token = args.tokenizer.eos_token

    # Load the model if necessary
    if args.bias=='direct':
        if args.graph_type.startswith('nback') or args.graph_type.startswith('tree'):
            pretrained_model = AutoModelForCausalLM.from_pretrained(f'{args.base_dir}/models/{args.pretrained_model_name}/best')
        else:
            pretrained_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, cache_dir=args.cache_dir)
    else:
        pretrained_model = None

    dataset, data_collator, val_loaders, tst_loaders = get_data_loaders(args)

    # Load the model
    config = load_config(args.model_type,args)
    model = AutoModelForCausalLM.from_config(config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)

    # Train the faiss index
    if args.graph_type.startswith('tree'):
        index_list, xb_list = calc_faiss_index(args)

    # Create the optimizer and the scheduler
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = gen_scheduler(optimizer, args)

    step_id = 0
    val_main_loss, val_attn_loss = evaluate(model,val_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'validation/val-main-{i+1}':loss
                    for i, loss in enumerate(val_main_loss)},step=step_id)
    wandb.log(data={f'validation/val-attn-{i+1}':loss
                    for i, loss in enumerate(val_attn_loss)},step=step_id)
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")

    best_val_loss = np.inf
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
                with torch.no_grad:
                    outputs_pretrained = pretrained_model(input_ids=loaded_examples['input_ids'],
                                                          labels=loaded_examples['labels'],
                                                          attention_mask=loaded_examples['attention_mask'],
                                                          output_attentions=True)
                attn_loss = torch.mean(torch.stack([torch.mean(torch.sum((attn1-attn2)**2,dim=(1,2,3)),dim=0)
                                                    for attn1, attn2 in zip(outputs.attentions, outputs_pretrained.attentions)]))
            else:
                layer_ids = np.arange(args.num_layers) if args.bias.split('-')[2]=='all' else [int(args.bias.split('-')[2])]
                if args.graph_type.startswith('nback'):
                    templates = get_templates(args, seq_len, batch_size, num_heads)
                    attn_loss = calc_attn_loss_nback(outputs.attentions, templates, layer_ids)
                elif args.graph_type.startswith('tree'):
                    attn_loss = calc_attn_loss_faiss(args, index_list, xb_list, outputs.attentions, layer_ids)

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
            model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")
            val_loss = np.mean(val_main_loss)+args.beta*np.mean(val_attn_loss)
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
                print(f'new best val loss: {best_val_loss}')
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/last")
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'test-last/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-last/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})

    model = AutoModelForCausalLM.from_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
    model.to(args.device)
    tst_main_loss, tst_attn_loss = evaluate(model,tst_loaders,args,pretrained_model=pretrained_model)
    wandb.log(data={f'test-best/tst-main-{i+1}':loss for i, loss in enumerate(tst_main_loss)})
    wandb.log(data={f'test-best/tst-attn-{i+1}':loss for i, loss in enumerate(tst_attn_loss)})
