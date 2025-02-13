import numpy as np
import torch
import glob
import faiss
from multiprocessing import Pool
import time
import math
import pickle
#import ot
from lsldg import calc_psi

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
    return np.load(path).astype('float32')

def load_attns(args, pca=False):
    if pca:
        attns = []
        for layer_id in range(args.num_layers):
            attns_layer = []
            for head_id in range(args.num_heads):
                print(f'Loading File {layer_id}-{head_id}')
                attns_layer.append(np.load(f'{args.base_dir}/attns/prep/{args.pretrained_model_name}/attns_{layer_id}_{head_id}.npy'))
            attns_layer = np.stack(attns_layer, axis=0)
            attns.append(attns_layer)
        attns = np.stack(attns, axis=0)
        assert len(attns.shape)==4, f"attns has an expected shape, {attns.shape}."
    else:
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
    #xb = xb.reshape((xb.shape[0]*xb.shape[1],xb.shape[2]*xb.shape[3]))
    rand_ids = args.rng.permutation(xb.shape[0])
    xb = xb[rand_ids]
    xb = xb.astype('float32')
    _, d = xb.shape

    print('Creating Index')
    start = time.time()
    #if args.graph_type in ['nback','tree']:
    #    red_dim = 256
    #else:
    #    red_dim = 512
    faiss_index = faiss.index_factory(d, f"HNSW,Flat")
    #faiss_index = faiss.index_factory(d, f"OPQ16_64,IVF{args.nlist},PQ16x4fsr")
    #faiss_index = faiss.index_cpu_to_gpus_list(faiss_index, gpus=[1,2,3])
    faiss_index.train(xb)
    faiss_index.add(xb)
    #faiss_index.nprobe = args.nprobe
    print(f'{time.time()-start}')
    return faiss_index, xb

def calc_faiss_index(args):
    attns = load_attns(args, pca=True)
    #attns = adjust_layer_assignment(attns, args.num_layers) # consumes too much memory
    # attns.shape = (batch_size, nlayers, nheads, seqlen, seqlen)
    #assert len(attns.shape)==5
    # attns.shape = (nlayers, batch_size, nheads, 512)
    assert len(attns.shape)==4
    index_list = []
    xb_list = []
    for layer_id in range(args.num_layers):
        index_list_layer = []
        xb_list_layer = []
        for head_id in range(args.num_heads):
            faiss_index, xb = create_index_job(attns[layer_id][head_id], args)
            index_list_layer.append(faiss_index)
            xb_list_layer.append(xb)
        index_list.append(index_list_layer)
        xb_list.append(xb_list_layer)
    return index_list, xb_list

def calc_attn_loss_faiss(args, pca_comps, index_list, xb_list, attns, layer_ids):
    # xb.shape[0]=5000*20, nlayers=8, nheads=4, seqlen=64
    # attns.shape = (nlayers, batchsize, nheads, seqlen, seqlen)
    attn_loss = 0
    for layer_id in layer_ids:
        for head_id in range(args.num_heads):
            attn = attns[layer_id][:,head_id,:,:] # attention matrices in the model that is being trained
            faiss_index = index_list[layer_id][head_id] # trained faiss_index
            xb = xb_list[layer_id][head_id] # bag of attention matrices from the pretrained model
            pca_comp = pca_comps[layer_id][head_id] # pca components
            assert len(attn.shape)==3 # batchsize, seqlen, seqlen

            # create no_grad version of attn to query the faiss_index
            xq = attn.clone().detach().cpu().numpy().astype('float32')
            xq = xq.reshape((xq.shape[0],xq.shape[1]*xq.shape[2])) # batchsize, seqlen*seqlen

            # reshape attn to match xb
            attn = attn.reshape((attn.shape[0],attn.shape[1]*attn.shape[2])) # batchsize, seqlen*seqlen
            attn = attn@pca_comp - attn.mean(axis=0)@pca_comp # project to pca comps

            _, I = faiss_index.search(xq, args.nneighbors)
            # xn.shape = (batchsize, neighbors, seqlen*seqlen)
            xn = torch.tensor(np.array([[xb[sample_id] for sample_id in line] for line in I]),device=args.device)
            assert len(xn.shape)==3 and xn.shape[1]==args.nneighbors

            if args.version=='central':
                dist = calc_centrality_dist(attn, xn)
            elif args.version=='softmin':
                dist = calc_softmin_dist(attn, xn)
            elif args.version.startswith('klestimate'):
                dist = calc_klestimate(attn, xn, args.knn)
            elif args.version.startswith('kde'):
                dist = calc_kde(attn, xn, args.sigma)
            attn_loss += torch.mean(dist)
    attn_loss /= len(layer_ids)*args.num_heads
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
    assert len(xn.shape)==3 and xn.shape[1]>=k and len(attn.shape)==2
    dist_inter = torch.sqrt(((attn-xn[:,k-1,:])**2).sum(dim=-1)+1e-10) # avoid numerical error by adding 1e-10
    samples = attn.detach().clone()
    dist_samples = torch.sqrt(torch.sum((samples.unsqueeze(1)-samples.unsqueeze(0))**2,dim=-1))
    knn_ids = torch.argsort(dist_samples,dim=-1)[:,k-1]
    knn_samples = torch.stack([samples[sample_id] for sample_id in knn_ids])
    dist_intra = torch.sqrt(((attn-knn_samples)**2).sum(dim=-1)+1e-10) # avoid numerical error by adding 1e-10
    return torch.log(dist_inter) - torch.log(dist_intra)

def calc_kde(attn,xn,sigma=1.0):
    assert len(xn.shape)==3 and len(attn.shape)==2
    dist = ((attn.unsqueeze(1)-xn)**2).sum(dim=-1)
    return -torch.exp(-dist/(2*sigma**2)).sum(dim=-1)

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

def calc_attn_loss_lsldg(args, pca_comps, centers, thetas, sigmas, attns, layer_ids):
    attn_loss = 0
    for layer_id in layer_ids:
        attn = attns[layer_id]
        assert len(attn.shape)==4
        attn = attn.reshape((attn.shape[0],attn.shape[1],attn.shape[2]*attn.shape[3]))
        attn = attn.reshape((attn.shape[0]*attn.shape[1],attn.shape[2]))
        attn = attn@pca_comps[layer_id] - attn.mean(axis=0)@pca_comps[layer_id] # project to pca comps

        psi = calc_psi(attn.detach().to('cpu').numpy(), centers[layer_id], sigmas[layer_id]) # nbases, nsamples, ndim
        psi = psi.transpose(2,1,0) # ndim, nsamples, nbases
        pgrad = np.einsum('bmn,bn->bm', psi, thetas[layer_id]) # ndim, nsamples
        pgrad = torch.tensor(pgrad.T, device=args.device)

        attn_loss += torch.mean(torch.sum(pgrad * attn, dim=-1))
    return attn_loss

def load_pca_comp(args, layer_id, head_id):
    with open(f'{args.base_dir}/pca/{args.pretrained_model_name}/pca_{layer_id}_{head_id}.pkl','rb') as f:
        pca = pickle.load(f)
    return torch.tensor(pca.components_.T,device=args.device)

def load_lsldg(args, layer_id):
    loaded_dict = np.load(f'{args.base_dir}/lsldg/{args.num_bases}bases/layer{layer_id}/best.npz')
    center = torch.tensor(loaded_dict['centers'], device=args.device)
    theta = torch.tensor(loaded_dict['theta'], device=args.device)
    sigma = loaded_dict['sig']
    return center, theta, sigma