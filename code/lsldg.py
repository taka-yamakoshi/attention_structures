import numpy as np
import torch
import math
from typing import List
import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import KFold
from utils_attn_loss import load_attns

def gauss_exp(x:np.ndarray, c:np.ndarray, sig:float):
    assert len(x.shape)==2 and len(c.shape)==2
    return np.exp(-np.linalg.norm(x[None,:,:]-c[:,None,:],axis=-1)**2/(2*sig**2))

def calc_psi(x:np.ndarray, c:np.ndarray, sig:float):
    return (c[:,None,:]-x[None,:,:])/(sig**2)*gauss_exp(x,c,sig)[:,:,None]

def calc_gh(x:np.ndarray, c:np.ndarray,
            sig:float,
            device:torch.device):
    psi = calc_psi(x,c,sig)
    print(psi.shape)
    assert len(psi.shape)==3
    num_bases, num_samples, ndim = psi.shape
    phi = -gauss_exp(x,c,sig)[:,:,None]/(sig**2) + ((c[:,None,:]-x[None,:,:])/(sig**2))*psi
    print(phi.shape)
    assert phi.shape[0]==num_bases and phi.shape[1]==num_samples and phi.shape[2]==ndim
    h = phi.mean(axis=1)
    bsize = 32
    bnum = math.ceil(ndim/bsize)
    g = []
    for i in range(bnum):
        tpsi = torch.tensor(psi[:,:,bsize*i:bsize*(i+1)]).to(device)
        print(tpsi.shape)
        g.append((tpsi.permute(2,0,1)@tpsi.permute(2,1,0)).to('cpu').numpy())
    g = np.concatenate(g)
    g /= num_samples
    return g, h

def solve_lsldg(x:np.ndarray, c:np.ndarray, 
                sig:float, lam:float, 
                device:torch.device):
    g, h = calc_gh(x, c, sig, device)
    theta = -np.linalg.solve(g + lam*np.eye(g.shape[1],dtype='float'),h.T)
    return theta

def run_cv(x:np.ndarray, c:np.ndarray,
           sig:float, lam:float,
           device:torch.device, nfold:int=5, seed:int=0):
    loss_list = []
    kf = KFold(n_splits=nfold,shuffle=True,random_state=seed)
    for trn_ids, tst_ids in kf.split(x):
        xtrn = x[trn_ids]
        xtst = x[tst_ids]
        theta = solve_lsldg(xtrn, c, sig, lam, device)
        gtst, htst = calc_gh(xtst, c, sig, device)
        term1 = (theta*(np.einsum('bmn,bnk->bmk', gtst, theta[:,:,None]).squeeze(-1))).sum(axis=1)
        term2 = (theta*(htst.T)).sum(axis=1)
        loss = term1 + 2*term2
        loss_list.append(loss.mean())
    return np.mean(loss_list)

def run(x:np.ndarray, num_bases:int, sigs:List, lams:List, device:torch.device, nfold:int=5):
    csv_data = []
    for seed in [100,200,300,400,500]:
        rng = np.random.RandomState(seed)
        rand_ids = rng.permutation(len(x))[:num_bases]
        c = x[rand_ids]
        for sig in sigs:
            for lam in lams:
                start = time.time()
                print(f'Running Seed{seed} Sigma{sig} Lambda{lam}')
                loss = run_cv(x, c, sig, lam, device, nfold=nfold, seed=seed+10)
                csv_data.append([sig,lam,seed,loss])
                print(time.time()-start)
    df = pd.DataFrame(csv_data,columns=['sig', 'lam', 'seed', 'loss'])
    #df_min = df.groupby(['sig', 'lam']).mean().reset_index().sort_values('loss').head(1)
    df_min = df.sort_values('loss').head(1)
    sig = df_min.sig.values[0]
    lam = df_min.lam.values[0]
    seed = df_min.seed.values[0]
    rng = np.random.RandomState(seed)
    rand_ids = rng.permutation(len(x))[:num_bases]
    c = x[rand_ids]
    theta = solve_lsldg(x, c, sig, lam, device)
    return df, sig, lam, seed, theta, rand_ids, c

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type = str, required=True)
    parser.add_argument('--num_layers', type = int, default = 8)
    parser.add_argument('--core_id', type = int, default = 0)
    parser.add_argument('--num_bases', type = int, default = 100)
    parser.add_argument('--nfold', type = int, default = 5)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")

    # Set the device
    args.device = torch.device("cuda", index=int(args.core_id)) if torch.cuda.is_available() else torch.device("cpu")

    # Set hyperparam search space
    sigs = [0.01,0.1,1.0,10]
    lams = [0.01,0.1,1.0,10]

    # Load data
    x = load_attns(args, pca=True)
    print(x.shape)
    for layer_id in range(args.num_layers):
        os.makedirs(f'{args.base_dir}/lsldg/{args.num_bases}bases/layer{layer_id}', exist_ok=True)
        df, sig, lam, seed, theta, rand_ids, c =  run(x[layer_id], args.num_bases, sigs, lams, args.device, args.nfold)
        df.to_csv(f'{args.base_dir}/lsldg/{args.num_bases}bases/layer{layer_id}/cv_results.csv', index=False)
        np.savez(f'{args.base_dir}/lsldg/{args.num_bases}bases/layer{layer_id}/best.npz',sig=sig,lam=lam,theta=theta,rand_ids=rand_ids,centers=c,x=x)
