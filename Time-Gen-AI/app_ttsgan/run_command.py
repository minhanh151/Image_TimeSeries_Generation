import subprocess 

def train(dataset_path: str, data_name: str, seq_len: int, sample_len: int,
          epochs: int, outdir: str, n_sample: int, iterations: int,
          ):
    command = [
        "python", "/app/main.py",
         "--model", "ttsgan",
        "--data_path", dataset_path,
        "--data_name", data_name,
        "--seq_len", str(seq_len),
        "--tts_max_epoch", str(epochs),
        "--n_samples", str(n_sample),
        "--tts_max_iter", str(iterations),
        "--sample_len", str(sample_len),
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