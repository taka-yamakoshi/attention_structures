import numpy as np
import torch
import glob
import faiss
from multiprocessing import Pool
import time
import math
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
        with Pool(processes=4) as p:
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
    start = time.time()
    if args.graph_type in ['nback','tree']:
        red_dim = 256
    else:
        red_dim = 512
    faiss_index = faiss.index_factory(d, f"PCA{red_dim},HNSW,Flat")
    #faiss_index = faiss.index_factory(d, f"OPQ16_64,IVF{args.nlist},PQ16x4fsr")
    #faiss_index = faiss.index_cpu_to_gpus_list(faiss_index, gpus=[1,2,3])
    faiss_index.train(xb)
    faiss_index.add(xb)
    #faiss_index.nprobe = args.nprobe
    print(f'{time.time()-start}')
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
    # xb.shape[0]=5000*20, nlayers=8, nheads=4, seqlen=64
    # attns.shape = (nlayers, batchsize, nheads, seqlen, seqlen)
    attn_loss = 0
    for layer_id in layer_ids:
        attn = attns[layer_id] # attention matrices in the model that is being trained
        faiss_index = index_list[layer_id] # trained faiss_index
        xb = xb_list[layer_id] # bag of attention matrices from the pretrained model
        assert len(attn.shape)==4 # batchsize, nheads, seqlen, seqlen

        # create no_grad version of attn to query the faiss_index
        xq = attn.clone().detach().cpu().numpy().astype('float32')
        xq = xq.reshape((xq.shape[0],xq.shape[1],xq.shape[2]*xq.shape[3])) # batchsize, nheads, seqlen*seqlen
        xq = xq.reshape((xq.shape[0]*xq.shape[1],xq.shape[2])) # batchsize*nheads, seqlen*seqlen

        # reshape attn to match xb
        attn = attn.reshape((attn.shape[0],attn.shape[1],attn.shape[2]*attn.shape[3])) # batchsize, nheads, seqlen*seqlen
        attn = attn.reshape((attn.shape[0]*attn.shape[1],attn.shape[2])) # batchsize*nheads, seqlen*seqlen

        _, I = faiss_index.search(xq, args.nneighbors)
        # xn.shape = (batchsize*nheads, neighbors, seqlen*seqlen)
        xn = torch.tensor(np.array([[xb[sample_id] for sample_id in line] for line in I]),device=args.device)
        assert len(xn.shape)==3 and xn.shape[1]==args.nneighbors

        if args.version=='central':
            dist = calc_centrality_dist(attn, xn)
        elif args.version=='softmin':
            dist = calc_softmin_dist(attn, xn)
        elif args.version=='klestimate':
            dist = calc_klestimate(attn, xn)
        attn_loss += torch.mean(dist)
    return attn_loss

def calc_centrality_dist(attn, xn):
    # instead of comparing to all the neighbors, identify the single "central" neighbor
    # center_vecs.shape = (batchsize*nheads, seqlen*seqlen)
    center_vecs = calc_graph_centers(xn)
    assert attn.shape[0]==xn.shape[0] and attn.shape[1]==xn.shape[2]
    return torch.sqrt(torch.sum((attn-center_vecs)**2,dim=-1)+1e-10) # avoid numerical error by adding 1e-10

def calc_softmin_dist(attn, xn):
    # use all neighbors and calculate the softmin
    # dist.shape = (batchsize*nheads, neighbors)
    dist = torch.sqrt(((attn.unsqueeze(1)-xn)**2).sum(dim=-1)+1e-10) # avoid numerical error by adding 1e-10
    return -torch.logsumexp(-dist, dim=1)

def calc_klestimate(attn, xn, k=20):
    assert len(xn.shape)==3 and xn.shape[1]>=k
    dist_inter = torch.sqrt(((attn-xn[:,k-1,:])**2).sum(dim=-1)+1e-10) # avoid numerical error by adding 1e-10
    return None

def calc_graph_centers(xn):
    bsize = 10 # limit RAM size to ~1.5GB
    assert len(xn.shape)==3
    bnum = math.ceil(xn.shape[0]/bsize)
    center_ids = []
    for i in range(bnum):
        a = xn[bsize*i:bsize*(i+1),:,:]
        dist = torch.sum(torch.sqrt(torch.sum((a.unsqueeze(2)-a.unsqueeze(1))**2,dim=-1)),dim=-1)
        center_ids.append(dist.argmin(dim=-1))
    center_ids = torch.cat(center_ids)
    return torch.stack([xn[i][cid] for i, cid in enumerate(center_ids)])
