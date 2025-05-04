import subprocess 
import sys 
sys.path.append("/wor")
def train(model_dir: str, output_dir: str, train_data_dir: str, 
          validation_images: list, validation_prompts: list,
          max_train_steps: int = 15000,
          learning_rate: float = 1e-5,
          resolution: int = 256,
          train_batch_size: int = 4,
          gradient_accumulation_steps: int = 4,
          checkpointing_steps: int = 1000,
          conditioning_image_column: str = "condition"):
    
    command = [
        "accelerate", "launch",
        "--config_file", "/app/configs/accelerate.yaml",
        "--main_process_port", "18000",
        "/app/app/train_controlnet.py",
        "--pretrained_model_name_or_path", model_dir,
        "--output_dir", output_dir,
        "--train_data_dir", train_data_dir,
        "--resolution", str(resolution),
        "--learning_rate", str(learning_rate),
        "--train_batch_size", str(train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--gradient_checkpointing",
        "--set_grads_to_none",
        "--conditioning_image_column", conditioning_image_column,
        "--max_train_steps", str(max_train_steps),
        "--checkpointing_steps", str(checkpointing_steps)
    ]
    
    # Add validation images
    for image in validation_images:
        command.extend(["--validation_image", image])
    
    # Add validation prompts
    for prompt in validation_prompts:
        command.extend(["--validation_prompt", f"'{prompt}'"])

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
    train(
        model_dir="$MODEL_DIR",
        output_dir="$OUTPUT_DIR",
        train_data_dir="/workspace/src/datasets/PlantVillage",
        validation_images=[
            "/workspace/src/datasets/PlantVillage_seg/val/Pepper__bell___Bacterial_spot/02baf62e-11e2-4dde-97fb-e369b57d55d3___JR_B.Spot 8971.JPG",
            "/workspace/src/datasets/PlantVillage_seg/val/Tomato__Target_Spot/dbb50cba-a410-49af-a95f-ac53d5fc9af3___Com.G_TgS_FL 8295.JPG"
        ],
        validation_prompts=[
            "bell pepper leaf showing symptoms of bacterial leaf spot with small water-soaked lesions that turn dark brown",
            "tomato leaf exhibiting symptoms of Target Spot. On the leaves the disease starts as small circular to oval dark brown to black spots. These spots enlarge becoming oval to angular and are normally confined within the main veins of the leaflets"
        ],
        max_train_steps=15000,
        checkpointing_steps=1000
    )