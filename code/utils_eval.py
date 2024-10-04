import numpy as np
import torch

from utils_attn_loss import get_templates, calc_attn_loss_nback, calc_attn_loss_faiss
def evaluate(model, loaders, args, pretrained_model=None, index_list=None, xb_list=None):
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

def evaluate_linzen(model, loaders, args):
    df = eval_model_linzen('../colorlessgreenRNNs/data/linzen_testset',args.tokenizer,model,args.device,max_length=args.max_length)
    df_group = df.filter(['num_attr','acc']).groupby(['num_attr'],as_index=False).mean()
    return {f'eval/linzen_test_{num_attr}':df_group.loc[lambda d: d['num_attr']==str(num_attr)]['acc'].item() for num_attr in range(5)}