import numpy as np
import os
import argparse
from utils_attn_loss import load_attns

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
    print(attns.shape)
    assert len(attns.shape)==5 and attns.shape[3]==attns.shape[4]
    batchsize, nlayers, nheads, seq_len, _ = attns.shape
    assert nlayers==args.num_layers and nheads==args.num_heads

    os.makedirs(f'{args.base_dir}/attns/stats/{args.pretrained_model_name}/',exist_ok=True)
    for layer_id in range(args.num_layers):
        for head_id in range(args.num_heads):
            np.save(f'{args.base_dir}/attns/stats/{args.pretrained_model_name}/attns_{layer_id}_{head_id}_mean.npy',
                    attns[:,layer_id,head_id,:,:].mean(axis=0))
            np.save(f'{args.base_dir}/attns/stats/{args.pretrained_model_name}/attns_{layer_id}_{head_id}_stdv.npy',
                    attns[:,layer_id,head_id,:,:].std(axis=0))
            np.save(f'{args.base_dir}/attns/stats/{args.pretrained_model_name}/attns_{layer_id}_{head_id}_log_mean.npy',
                    np.log(attns[:,layer_id,head_id,:,:]+1e-10).mean(axis=0))
            np.save(f'{args.base_dir}/attns/stats/{args.pretrained_model_name}/attns_{layer_id}_{head_id}_log_stdv.npy',
                    np.log(attns[:,layer_id,head_id,:,:]+1e-10).std(axis=0))
