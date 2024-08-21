import numpy as np
import torch
import argparse
import os
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from utils import gen_dataset_name, gen_run_name_key_amp, seed_everything
from train import load_config, get_data_loaders, gen_scheduler

import sys
sys.path.append('../../pyvene')
import pyvene as pv

class KeyAmpConfig(PretrainedConfig):
    model_type = 'llama'
    is_composition = True
    def __init__(self, **kwargs):
        self.alpha = kwargs.pop('alpha')
        self.layer_id = kwargs.pop('layer_id')
        super().__init__(**kwargs)
    def to_dict(self):
        output = super().to_dict()
        output['alpha'] = self.alpha
        output['layer_id'] = self.layer_id
        return output

class KeyAmpModel(PreTrainedModel):
    config_class = KeyAmpConfig
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        config_kwargs = config.to_dict()
        self.alpha = config_kwargs.pop('alpha')
        self.layer_id = config_kwargs.pop('layer_id')
        main_config = LlamaConfig(**config_kwargs)
        main_model = AutoModelForCausalLM.from_config(main_config)
        self.pv_model = pv.IntervenableModel(
            {"component": f"model.layers[{self.layer_id}].self_attn.k_proj.output",
            "intervention": self.interv_fn},
             model=main_model)
        self.pv_model.enable_model_gradients()
    def interv_fn(self,base,sources):
        #mask = (self.input_ids>=23)*self.alpha + (self.input_ids<23)
        mask = self.alpha*self.surprisal
        return mask.unsqueeze(-1)*base
    def calc_surprisal(self, logits:torch.Tensor, labels:torch.Tensor):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_labels = shift_labels.to(shift_logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        surprisal = (1/self.alpha)*torch.ones(labels.shape).float()
        for bid, (batch_logits, batch_labels) in enumerate(zip(shift_logits, shift_labels)):
            surprisal[bid][1:] = loss_fct(batch_logits, batch_labels)
        return surprisal
    def resize_token_embeddings(self, size:int):
        self.pv_model.model.resize_token_embeddings(size)
    def forward(
        self,
        examples,
        **kwargs,
        ):
        examples.update(kwargs)
        self.input_ids = examples['input_ids']
        with torch.no_grad():
            outputs = self.pv_model.model(**examples)
        self.surprisal = self.calc_surprisal(outputs.logits, examples['labels'])
        orgnl_outputs, interv_outputs = self.pv_model(base=examples,
                                                      output_original_output=True)
        return orgnl_outputs, interv_outputs

def evaluate(model, loaders, args):
    out = []
    model.eval()
    with torch.no_grad():
        for loader in loaders:
            loss_list = []
            for examples in loader:
                loaded_examples = examples.to(args.device)
                _, interv_outputs = model(loaded_examples, output_attentions=True)
                loss_list.append(interv_outputs.loss.item())
            out.append(np.mean(loss_list))
    return out

if __name__=='__main__':
    parser = argparse.ArgumentParser()
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
    
    parser.add_argument('--alpha', type = float, default = 1.0)
    parser.add_argument('--layer_id', type = int, default = 0)

    parser.add_argument('--batchsize_trn', type = int, default = 10)
    parser.add_argument('--batchsize_val', type = int, default = 100)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--scheduler_type', type = str, default = 'constant')
    parser.add_argument('--num_epochs', type = int, default = 5)
    parser.add_argument('--run_seed', type = int, default = 1234)

    parser.add_argument('--core_id', type = int, default = 0)
    args = parser.parse_args()
    print(f'running with {args}')

    # Initialize weights and biases with args
    import wandb
    wandb.require("core")
    wandb.init(project="attn_struct_key_amp_surprisal")
    wandb.config.update(args.__dict__)

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id))

    # Generate the dataset name and the run name
    args.dataset_name = gen_dataset_name(args)
    args.run_name = gen_run_name_key_amp(args)
    wandb.config.dataset_name = args.dataset_name
    wandb.config.run_name = args.run_name

    # Fix the seed
    seed_everything(args.run_seed)

    # Load the tokenizer
    if args.graph_type.startswith('nback'):
        # Load the tokenizer for all n's
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_nback-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    elif args.graph_type.startswith('tree'):
        # Load the tokenizer for all trees
        args.tokenizer = AutoTokenizer.from_pretrained(f'{args.base_dir}/tokenizers/{args.model_type}_tree-all_{args.vocab_size}_{args.max_prob}_{args.seq_len}_{args.seed}')
    if args.model_type in ['gpt2','llama2']:
        args.tokenizer.pad_token = args.tokenizer.eos_token

    # Load the dataset
    dataset, data_collator, val_loaders, tst_loaders = get_data_loaders(args)

    # Load the model
    config = load_config(args.model_type,args)
    config.alpha = args.alpha
    config.layer_id = args.layer_id
    config = KeyAmpConfig(**(config.to_dict()))
    model = KeyAmpModel(config)
    model.resize_token_embeddings(len(args.tokenizer))
    model.to(args.device)

    # Create the optimizer and the scheduler
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    scheduler = gen_scheduler(optimizer, args)

    step_id = 0
    val_loss = evaluate(model,val_loaders,args)
    wandb.log(data={f'validation/val-{i+1}':loss for i, loss in enumerate(val_loss)}, step=step_id)
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
            _, interv_outputs = model(loaded_examples, output_attentions=True)

            loss = interv_outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            step_id += 1
            lr = scheduler.get_last_lr()[0]
            if step_id%100==0:
                wandb.log(data={'train/lr':lr, 'train/loss':loss.item()}, step=step_id)
        if epoch%(max(args.num_epochs//10,1))==0:
            val_loss = evaluate(model,val_loaders,args)
            wandb.log(data={f'validation/val-{i+1}':loss for i, loss in enumerate(val_loss)}, step=step_id)
            model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/ckpt-{step_id}")
            val_loss = np.mean(val_loss)
            if val_loss<best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
                print(f'new best val loss: {best_val_loss}')
    model.save_pretrained(f"{args.base_dir}/models/{args.run_name}/last")
    tst_loss = evaluate(model,tst_loaders,args)
    wandb.log(data={f'test-last/tst-{i+1}':loss for i, loss in enumerate(tst_loss)})

    model = KeyAmpModel.from_pretrained(f"{args.base_dir}/models/{args.run_name}/best")
    model.to(args.device)
    tst_loss = evaluate(model,tst_loaders,args)
    wandb.log(data={f'test-best/tst-{i+1}':loss for i, loss in enumerate(tst_loss)})
