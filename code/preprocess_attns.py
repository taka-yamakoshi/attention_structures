import numpy as np
import os
import argparse
import pickle
from utils_attn_loss import load_attns
from sklearn.decomposition import PCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name', type = str, default='llama2_babylm_5_0.8_16_1234_8_4_512_2048_nobias_0.0_10000000_32_500_0.0001_constant_5_1000')
    parser.add_argument('--graph_type', type = str, default='babylm')
    parser.add_argument('--num_layers', type = int, default = 8)
    parser.add_argument('--num_heads', type = int, default = 4)
    args = parser.parse_args()
    print(f'running with {args}')

    # Set the storage path
    args.base_dir = os.environ.get("MY_DATA_PATH")
    args.cache_dir = f'{args.base_dir}/cache'
    os.makedirs(args.cache_dir, exist_ok=True)

    attns = load_attns(args)
    #attns = adjust_layer_assignment(attns, args.num_layers)

    print(attns.shape)
    assert len(attns.shape)==5 and attns.shape[3]==attns.shape[4]
    batchsize, nlayers, nheads, seq_len, _ = attns.shape
    assert nlayers==args.num_layers and nheads==args.num_heads

    os.makedirs(f'{args.base_dir}/attns/prep/{args.pretrained_model_name}/',exist_ok=True)
    os.makedirs(f'{args.base_dir}/pca/{args.pretrained_model_name}',exist_ok=True)
    for layer_id in range(args.num_layers):
        pca = PCA(n_components=512)
        attns_new = pca.fit_transform(attns[:,layer_id,:,:,:].reshape(batchsize,nheads,seq_len*seq_len).reshape(batchsize*nheads,seq_len*seq_len))
        np.save(f'{args.base_dir}/attns/prep/{args.pretrained_model_name}/attns_{layer_id}.npy',attns_new)
        with open(f'{args.base_dir}/pca/{args.pretrained_model_name}/pca_{layer_id}.pkl','wb') as f:
            pickle.dump(pca,f,protocol=5)
