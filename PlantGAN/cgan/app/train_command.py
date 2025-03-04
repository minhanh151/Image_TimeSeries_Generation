import subprocess 
import sys 
sys.path.append("/wor")
def train(dataset_path: str, outdir: str, epochs: int, 
          batch_size: int, input_size: int, class_num: int,
          ):
    command = [
        "python", "/app/train.py",
        "--dataroot", dataset_path,
        "--model_mode", "2",
        "--name", "cgan",
        "--model", "cgan",
        "--epoch_count", str(epochs),
        "--batch_size", str(batch_size),
        "--input_size", str(input_size),
        "--class_num", str(class_num),
        "--name", outdir
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


if __name__ == "__main__":
    pass