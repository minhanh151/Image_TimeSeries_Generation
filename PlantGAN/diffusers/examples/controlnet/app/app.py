import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import zipfile
import shutil
from train import train
from inference import inference
import glob
# from infer import generate

app = FastAPI()
MODEL_TYPE = os.getenv("MODEL_TYPE", "default")

def remove_file(file_path: str):
    try:
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        else:
            os.remove(file_path)
    except Exception as e:
        print(f"Error removing file: {e}")


@app.post("/fine-tune-and-generate")
async def main(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    condition_images: UploadFile = File(...),
    model_type: str = Form(...),
    n_infer_steps: int = Form(10),
    steps: int = Form(3000),
    atten_scale: float = Form(0.8),
    guidance_scale: float = Form(7.5),
    n_sample: int = Form(1),
):
    if model_type != MODEL_TYPE:
        raise HTTPException(400, f"Model type {model_type} not supported by this container")
    
    # Process dataset
    tmp_dir = tempfile.mkdtemp()
    
    # Save uploaded dataset file
    dataset_path = f"{tmp_dir}/dataset.zip"
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    
    # Extract dataset
    extracted_dataset_dir = f"{tmp_dir}/dataset"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dataset_dir)
    
    # Save uploaded condition images file
    condition_images_path = f"{tmp_dir}/condition_images.zip"
    with open(condition_images_path, "wb") as f:
        f.write(await condition_images.read())
    
    # Extract condition images
    extracted_condition_dir = f"{tmp_dir}/condition_images"
    with zipfile.ZipFile(condition_images_path, "r") as zip_ref:
        zip_ref.extractall(extracted_condition_dir)
    
    # Find some validation images for training
    validation_images = glob.glob(f"{extracted_condition_dir}/*/*.jpg") or glob.glob(f"{extracted_condition_dir}/*/*.png")
    if not validation_images:
        validation_images = glob.glob(f"{extracted_condition_dir}/*.jpg") or glob.glob(f"{extracted_condition_dir}/*.png")
    
    # Use the first few images as validation images (limited to 2 for simplicity)
    validation_images = validation_images[:2] if len(validation_images) >= 2 else validation_images
    
    # Default validation prompts
    validation_prompts = [
        "plant leaf with visible signs of disease",
        "plant leaf showing symptoms of disease"
    ]
    
    # Create output model directory
    model_output_dir = f"{tmp_dir}/model"
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Train model with corrected parameters
    print("training model")
    train(
        model_dir="runwayml/stable-diffusion-v1-5",  # Use default SD model
        output_dir=model_output_dir,
        train_data_dir=extracted_dataset_dir,
        validation_images=validation_images,
        validation_prompts=validation_prompts,
        max_train_steps=steps
    )
    
    # Generate data
    print("generating data")
    inference(model=model_output_dir, 
              folder_save=f"{tmp_dir}/output", 
              folder_condition=extracted_condition_dir,
              n_infer_steps=n_infer_steps, 
              atten_scale=atten_scale, 
              guidance_scale=guidance_scale,
              num_sample=n_sample)
    
    output_images = glob.glob(f"{tmp_dir}/output/*.png")
    output_zip = f"{tmp_dir}/output.zip"
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for idx, img_path in enumerate(output_images):
            zipf.write(img_path, arcname=f"result_{idx}.png")

    # Schedule cleanup for temporary directory
    background_tasks.add_task(remove_file, tmp_dir)
    return FileResponse(output_zip, filename="output.zip")