import subprocess 

def train(dataset_path: str, data_name: str, seq_len: int, 
          epochs: int, outdir: str, n_sample: int
          ):
    command = [
        "python", "/app/main.py",
        "--data_path", dataset_path,
        "--data_name", data_name,
        "--seq_len", seq_len,
        "--rts_epochs", epochs,
        "--n_smaples", n_sample,
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


# if __name__ == "__main__":
#     train(dataset_path)