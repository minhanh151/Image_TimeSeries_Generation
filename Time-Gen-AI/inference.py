## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import argparse
from config.config import parse_args
from models.missingprocessor import Processor
import numpy as np 
import pandas as pd
import torch
import os
# ==================TimeGAN ================================
from models.timegan import timegan

# ====================== RTSGAN ===================================
from models.aegan import AeGAN
from utils import init_logger

# ===================== DGAN ======================================
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

# ==================== TTSGAN ====================================
from models.GANModels import Generator, Discriminator
from models.ttsgan import train_ttstgan

from data_loading import real_data_loading, loading_RTS_dataset, stock_dataset
from metrics.visualization_metrics import visualization


cols = ['Open','High','Low','Close','Adj_Close','Volume']
def create_list_of_row_arrays_v2(df, target_length=24):
    """
    Groups the DataFrame by 'example_id' and creates a list of NumPy arrays
    of a fixed specified length. Each array contains the values from all
    columns *except* 'example_id' for the rows within each group.
    Assumes each group will have exactly 'target_length' rows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_length (int): The desired and expected number of rows in each group.

    Returns:
        list: A list of NumPy arrays.
    """
    result_list = []
    for example_id, group in df.groupby('example_id'):
        if len(group) != target_length:
            raise ValueError(f"Group with example_id '{example_id}' has {len(group)} rows, expected {target_length}.")
        # Select all columns except 'example_id' and convert to NumPy array
        row_array = group.drop(columns=['example_id']).to_numpy()
        result_list.append(row_array)
    return result_list

def np2csv(seq_len: np.ndarray, cols) -> pd.DataFrame:
    dfs = []
    # Iterate through each sample in seq_len
    for i in range(seq_len.shape[0]):
        sample_data = seq_len[i, :, :]  # Extract data for the current sample
        sample_df = pd.DataFrame(sample_data, columns=cols)
        sample_df['sample_id'] = i  # Add a column for the sample ID
        dfs.append(sample_df)

    # Concatenate the list of DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    
    return final_df
if __name__ == '__main__':
    args = parse_args() 

    root_dir = "{}/{}".format(args.log_dir, args.data_name)
  
    logger = init_logger(root_dir)

    params=vars(args)
  
    params["root_dir"]= root_dir
    params["logger"]= logger
    params["device"]= 'cuda:{}'.format(args.device)
    
    params['iterations'] = args.iteration # for rts gan, timegan 
    params['batch_size'] = args.batch_size
    
    print(params.keys())
    
    ori_data = real_data_loading(args.data_name, args.seq_len)

    ## ============ Data loading ========================
    # rtsgan and timegan both use tensorflow while ttsgan and dgan use pytorch
    if args.model == 'rtsgan':
        dataset = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len, return_dict=True)
        train_set= dataset["train_set"]
        dynamic_processor= dataset["dynamic_processor"]
        static_processor= dataset["static_processor"]
        train_set.set_input("sta","dyn","seq_len")
    elif args.model == 'timegan' or args.model == 'doppelgan':
        dataset = ori_data  
    elif args.model == 'ttsgan':
        dataset, processor = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len)
        dataset = stock_dataset(dataset)
    # define model architecture 
    if args.model == 'rtsgan':

        aegan = AeGAN((static_processor, dynamic_processor), params)
        aegan.load_generator(f'{args.model_path}/generator.dat')
        aegan.load_ae(f'{args.model_path}/ae.dat')
        generated_data = aegan.synthesize(args.n_samples)
        generated_data = [np.array(data) for data in generated_data]
    elif args.model == 'timegan': 
        params['module'] = args.timegan_module
        params['hidden_dim'] = args.timegan_hidden_dim
        params['num_layer'] = args.timegan_num_layer 
        generated_data = timegan (ori_data, params)
    elif args.model == 'doppelgan':
        config = DGANConfig(
        max_sequence_len=args.seq_len,
        sample_len=args.sample_len,
        batch_size=args.batch_size,
        epochs=10000
        )
        dgan = DGAN(config) 
        dgan = dgan.load(os.path.join(args.model_path, "dgan_df.pth"))
        generated_data_csv = dgan.generate_dataframe(args.n_samples)
        generated_data = create_list_of_row_arrays_v2(generated_data_csv)
    elif args.model == 'ttsgan':
        # import network
        gen_net = Generator(
            seq_len=args.seq_len, 
            patch_size=args.tts_patch_size, 
            channels=args.sample_len, 
            num_classes=1, 
            latent_dim=args.tts_latent_dim, 
        )
        checkpoint = torch.load(os.path.join(args.model_path, "tts_ckpt.pth"), map_location='cuda:0'.format(args.device))
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        gen_net.to('cuda:0'.format(args.device))

        generated_data = [] 

        for i in range(args.n_samples):
            fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to('cuda:0'.format(args.device))
            fake_sigs = gen_net(fake_noise).to('cpu').detach().numpy()
            fake_sigs = fake_sigs.squeeze().transpose(1,0)
            # print(fake_sigs.shape)
            generated_data.append(fake_sigs)
        generated_data = [processor.inverse_transform(data) for data in generated_data]
  
    # print(generated_data)
    if args.model == 'doppelgan':
        generated_data_csv.to_csv(f'{args.outdir}/data.csv')
    else:
        generated_data_csv = np2csv(np.array(generated_data), cols)
        generated_data_csv.to_csv(f'{args.outdir}/data.csv', index=False)
    
    # np.save(os.path.join(args.outdir, "output.npy"), generated_data)

     # 3. Visualization (PCA and tSNE)
    # print(generated_data[0].shape)
    
    visualization(ori_data[:args.n_samples], generated_data, 'pca')
    if args.n_samples > 40:
        visualization(ori_data[:args.n_samples], generated_data, 'tsne')