import subprocess 
import sys 
sys.path.append("/wor")
def train(dataset_path: str, outdir: str, steps: int,
          pretrained_path : str ="runwayml/stable-diffusion-v1-5",
          ):
    command = [
        "accelerate", "launch",
        "--main_process_port", "18000",
        "/app/app/train_text_to_image_lora.py",
        "--pretrained_model_name_or_path", pretrained_path,
        "--dataset_name", dataset_path,
        "--dataloader_num_workers", "8",
        "--resolution", "256",
        "--random_flip",
        "--train_batch_size", "4",
        "--gradient_accumulation_steps", "4", 
        "--max_train_steps", str(steps),
        "--learning_rate", "1e-04",
        "--max_grad_norm", "1",
        "--lr_scheduler","cosine",
        "--lr_warmup_steps", "0",
        "--output_dir", outdir,
        "--checkpointing_steps", "500",
        "--validation_prompt", "'tomato leaf Mosaic Virus with mottled patterns of light and dark green uneven leaf coloring and a general mosaic-like appearance'",
        "--resume_from_checkpoint", "latest",
        "--seed", "1337"

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
    train(dataset_path="/workspace/src/PlantGAN/datasets/test",
          outdir="plantVillage_text2image", steps=1)