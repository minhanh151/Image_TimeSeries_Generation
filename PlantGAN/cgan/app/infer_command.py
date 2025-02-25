import subprocess 

def infer(model: str, outdir: str, niter: int, batch: int,
          zdim_in: int, input_size: int, class_num: int,
          ):
    command = [
        "python", "/app/infer.py",
        "--weight", model,
        "--zdim_in", str(zdim_in),
        "--batch", str(batch),
        "--niter", str(niter),
        "--input_size", str(input_size),
        "--classes", str(class_num),
        "--path-save", outdir
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