import subprocess 
import sys 
sys.path.append("/wor")
def train(dataset_path: str, outdir: str, steps: int,
          ):
    command = [
        "python", "/app/train.py",
        "--data", dataset_path,
        "--outdir", outdir,
        "--cfg", "auto",
         "--cond", "1"
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
    train(dataset_path="/workspace/src/dataset",
          outdir="plantVillage_text2image", steps=1)