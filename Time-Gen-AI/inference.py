"""Main script for running inference with various GAN models."""

# Standard library imports
from __future__ import absolute_import, division, print_function
import os
from typing import List, Dict, Any

# Third-party imports
import numpy as np
import pandas as pd
import torch

# Local imports
from config.config import parse_args
from models.missingprocessor import Processor
from models.timegan import timegan
from models.aegan import AeGAN
from models.GANModels import Generator, Discriminator
from models.ttsgan import train_ttstgan
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig
from data_loading import real_data_loading, loading_RTS_dataset, stock_dataset
from metrics.visualization_metrics import visualization
from utils import init_logger

# Constants
STOCK_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']

def create_list_of_row_arrays(df: pd.DataFrame, target_length: int = 24) -> List[np.ndarray]:
    """Creates a list of fixed-length arrays from DataFrame grouped by example_id.
    
    Args:
        df: Input DataFrame containing time series data
        target_length: Expected number of rows in each group
        
    Returns:
        List of NumPy arrays containing time series data
        
    Raises:
        ValueError: If any group doesn't match the target length
    """
    result_list = []
    for example_id, group in df.groupby('example_id'):
        if len(group) != target_length:
            raise ValueError(
                f"Group with example_id '{example_id}' has {len(group)} rows, "
                f"expected {target_length}."
            )
        row_array = group.drop(columns=['example_id']).to_numpy()
        result_list.append(row_array)
    return result_list

def numpy_to_dataframe(seq_data: np.ndarray, columns: List[str]) -> pd.DataFrame:
    """Converts 3D numpy array of sequences to a DataFrame.
    
    Args:
        seq_data: Numpy array of shape (n_samples, seq_length, n_features)
        columns: Column names for the features
        
    Returns:
        DataFrame with samples concatenated vertically and sample_id column added
    """
    dfs = []
    for i in range(seq_data.shape[0]):
        sample_df = pd.DataFrame(seq_data[i, :, :], columns=columns)
        sample_df['sample_id'] = i
        dfs.append(sample_df)
    return pd.concat(dfs, ignore_index=True)

def setup_model_params(args) -> Dict[str, Any]:
    """Sets up model parameters and logging.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary containing model parameters
    """
    root_dir = f"{args.log_dir}/{args.data_name}"
    logger = init_logger(root_dir)
    
    params = vars(args)
    params.update({
        "root_dir": root_dir,
        "logger": logger,
        "device": f'cuda:{args.device}',
        "iterations": args.iteration,
        "batch_size": args.batch_size
    })
    return params

def generate_rtsgan_samples(dataset: Dict, params: Dict, args) -> List[np.ndarray]:
    """Generates samples using RTSGAN model."""
    train_set = dataset["train_set"]
    dynamic_processor = dataset["dynamic_processor"]
    static_processor = dataset["static_processor"]
    train_set.set_input("sta", "dyn", "seq_len")
    
    aegan = AeGAN((static_processor, dynamic_processor), params)
    aegan.load_generator(f'{args.model_path}/generator.dat')
    aegan.load_ae(f'{args.model_path}/ae.dat')
    
    generated_data = aegan.synthesize(args.n_samples)
    return [np.array(data) for data in generated_data]

def generate_doppelgan_samples(args) -> List[np.ndarray]:
    """Generates samples using DoppelGAN model."""
    config = DGANConfig(
        max_sequence_len=args.seq_len,
        sample_len=args.sample_len,
        batch_size=args.batch_size,
        epochs=10000
    )
    dgan = DGAN(config)
    dgan = dgan.load(os.path.join(args.model_path, "dgan_df.pth"))
    generated_data_csv = dgan.generate_dataframe(args.n_samples)
    # return 
    return create_list_of_row_arrays(generated_data_csv, args.seq_len), generated_data_csv

def generate_ttsgan_samples(processor, args) -> List[np.ndarray]:
    """Generates samples using TTSGAN model."""
    gen_net = Generator(
        seq_len=args.seq_len,
        patch_size=args.tts_patch_size,
        channels=args.sample_len,
        num_classes=1,
        latent_dim=args.tts_latent_dim,
    )
    device = f'cuda:{args.device}'
    checkpoint = torch.load(os.path.join(args.model_path, "tts_ckpt.pth"), map_location=device)
    gen_net.load_state_dict(checkpoint['gen_state_dict'])
    gen_net.to(device)

    generated_data = []
    for _ in range(args.n_samples):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100))).to(device)
        fake_sigs = gen_net(fake_noise).cpu().detach().numpy()
        fake_sigs = fake_sigs.squeeze().transpose(1, 0)
        generated_data.append(fake_sigs)
        
    return [processor.inverse_transform(data) for data in generated_data]

def main():
    """Main execution function."""
    args = parse_args()
    params = setup_model_params(args)
    
    # Load original data
    ori_data = real_data_loading(args.data_name, args.seq_len)
    
    # Load dataset based on model type
    if args.model == 'rtsgan':
        dataset = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len, return_dict=True)
        generated_data = generate_rtsgan_samples(dataset, params, args)
    elif args.model in ['timegan', 'doppelgan']:
        dataset = ori_data
        if args.model == 'timegan':
            params.update({
                'module': args.timegan_module,
                'hidden_dim': args.timegan_hidden_dim,
                'num_layer': args.timegan_num_layer
            })
            generated_data = timegan(ori_data, params)
        else:
            generated_data, generated_data_csv = generate_doppelgan_samples(args)
    elif args.model == 'ttsgan':
        dataset, processor = loading_RTS_dataset(args.data_path, args.data_name, args.seq_len)
        dataset = stock_dataset(dataset)
        generated_data = generate_ttsgan_samples(processor, args)
    
    # Save generated data
    if args.model != 'doppelgan':
        generated_data_csv = numpy_to_dataframe(np.array(generated_data), STOCK_COLUMNS)
    
    generated_data_csv.to_csv(f'{args.outdir}/data.csv', index=False)
    
    # Visualize results
    
    if args.n_samples > 40:
        visualization(ori_data[:args.n_samples], generated_data, 'pca')
        visualization(ori_data[:args.n_samples], generated_data, 'tsne')

if __name__ == '__main__':
    main()