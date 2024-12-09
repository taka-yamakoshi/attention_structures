import numpy as np
import torch
import glob
import faiss
from multiprocessing import Pool
#import ot

def get_template_nback(n:int, seq_len:int, batch_size:int, num_heads:int):
    # adjusted for next toke prediction
    mat = torch.tensor([[[[1 if j==i-(n-1) else 0
                           for j in range(seq_len)]
                           for i in range(seq_len)]
                           for _ in range(num_heads)]
                           for _ in range(batch_size)])
    return mat

def get_templates(args, seq_len, batch_size, num_heads):
    ns = [1,2,3,4,5] if args.bias.startswith('nback-all') else [int(args.bias.split('-')[1])]
    return torch.stack([get_template_nback(n,seq_len,batch_size,num_heads).to(args.device) for n in ns])

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

def load_attn_job(job_id, path):
    print(f"Loading File{job_id}")
    return np.load(path)

def load_attns(args):
    if args.graph_type.startswith('tree') or args.graph_type.startswith('nback'):
        attns_path = f'{args.base_dir}/attns/{args.pretrained_model_name}/attns.npy'
        attns = np.load(attns_path)
    else:
        pool_args = [(i, path) for i, path in enumerate(glob.glob(f'{args.base_dir}/attns/{args.pretrained_model_name}/attns_*.npy'))]
        with Pool(processes=16) as p:
            attns = p.starmap(load_attn_job,pool_args)
        attns = np.concatenate(attns, axis=0)
    print('Finished Loading')
    return attns

def create_index_job(xb, args):
    xb = xb.reshape((xb.shape[0]*xb.shape[1],xb.shape[2]*xb.shape[3]))
    rand_ids = args.rng.permutation(xb.shape[0])
    xb = xb[rand_ids]
    xb = xb.astype('float32')
    _, d = xb.shape

    print('Creating Index')
    if args.graph_type in ['nback','tree']:
        red_dim = 256
    else:
        red_dim = 512
    faiss_index = faiss.index_factory(d, f"PCA{red_dim},IVF{args.nlist},Flat")
    #faiss_index = faiss.index_factory(d, f"OPQ16_64,IVF{args.nlist},PQ16x4fsr")
    faiss_index.train(xb)
    faiss_index.add(xb)
    faiss_index.nprobe = args.nprobe
    return faiss_index, xb

def calc_faiss_index(args):
    attns = load_attns(args)
    #attns = adjust_layer_assignment(attns, args.num_layers) # consumes too much memory
    # attns.shape = (batch_size, nlayers, nheads, seqlen, seqlen)
    assert len(attns.shape)==5
    index_list = []
    xb_list = []
    for layer_id in range(args.num_layers):
        faiss_index, xb = create_index_job(attns[:,layer_id,:,:,:], args)
        index_list.append(faiss_index)
        xb_list.append(xb)
    return index_list, xb_list

def calc_attn_loss_faiss(args, index_list, xb_list, attns, layer_ids):
    # attns.shape = (nlayers, batchsize, nheads, seqlen, seqlen)
    attn_loss = 0
    for layer_id in layer_ids:
        attn = attns[layer_id]
        faiss_index = index_list[layer_id]
        print(faiss_index.nprobe)
        xb = xb_list[layer_id]
        assert len(attn.shape)==4
        xq = attn.clone().detach().cpu().numpy().astype('float32')
        xq = xq.reshape((xq.shape[0]*xq.shape[1],xq.shape[2]*xq.shape[3]))
        _, I = faiss_index.search(xq, args.nneighbors)

        # xn.shape = (batchsize*nheads, neighbors, seqlen*seqlen)
        xn = torch.tensor(np.array([[xb[sample_id] for sample_id in line] for line in I]),device=args.device)
        assert len(xn.shape)==3 and xn.shape[1]==args.nneighbors

        # attn.shape = (batchsize*nheads, seqlen*seqlen)
        attn = attn.view(attn.shape[0]*attn.shape[1],attn.shape[2]*attn.shape[3])
        assert attn.shape[0]==xn.shape[0] and attn.shape[1]==xn.shape[2]

        # dist.shape = (batchsize*nheads, neighbors)
        dist = torch.sqrt(((attn.unsqueeze(1)-xn)**2).sum(dim=-1)+1e-10)

        min_dist = -torch.logsumexp(-dist, dim=1)
        attn_loss += torch.mean(min_dist)
    return attn_loss