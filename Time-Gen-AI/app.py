import os
import subprocess
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image

def infer(
    model : str,
    model_path: str,
    dataset_path: str,
    data_name: str,
    seq_len: int,
    module: str,
    hidden_dim: int,
    num_layer: int,
    iteration: int,
    batch_size: int,
    outdir: str,
    n_samples: int,
):
    command = [
        "python", "inference.py",
        "--model", model,
        "--model_path", model_path,
        "--data_path", dataset_path,
        "--data_name", data_name,
        "--seq_len", str(seq_len),
        "--timegan_module", module,
        "--timegan_hidden_dim", str(hidden_dim),
        "--timegan_num_layer", str(num_layer),
        "--iteration", str(iteration), 
        "--batch_size", str(batch_size),
        "--n_samples", str(n_samples),
        "--outdir", outdir,
    ]
    # Run the .sh file
    result = subprocess.run(command, capture_output=True, text=True)

    # Print the output and errors
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    # Check the return code (0 means success)
    if result.returncode == 0:
        print("Script executed successfully!")
    else:
        print("Script failed with return code:", result.returncode)


st.title("Timeseries Generation Tool")

model_type: str = st.selectbox(
    "Select model", ("timegan", "rtsgan", "doppelgan", "ttsgan")
)
n_samples: int = st.number_input("Number of samples", min_value=2)

model_path : str = "weights"
data_path  : str = "data"
data_name  : str = "stock"
seq_len    : int = 24
module     : str = "gru"
hidden_dim : int = 34
num_layer  : int = 3
iteration  : int = 5
batch_size : int = 128
# n_samples  : int = 1

if st.button("Infer"):
    infer(
        model_type,
        model_path,
        data_path,
        data_name,
        seq_len,
        module,
        hidden_dim,
        num_layer,
        iteration,
        batch_size,
        "stock_result",
        n_samples,
    )
    # if model_type == "doppelgan":
    st.write(pd.read_csv(os.path.join("stock_result", "data.csv")))
    # else:
    #     st.write(np.load(os.path.join("stock_result", "output.npy")))
    
    
    st.image(Image.open(os.path.join("stock_result", "pca_plot.png")), caption="PCA Plot")
    if n_samples > 40:
        st.image(Image.open(os.path.join("stock_result", "tsne_plot.png")), caption="t-SNE Plot")