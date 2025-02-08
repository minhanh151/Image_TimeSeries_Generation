import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import zipfile
import shutil
from train_command import train
from infer_command import inference
import glob
# from infer import generate

app = FastAPI()
MODEL_TYPE = os.getenv("MODEL_TYPE", "default")

def remove_file(file_path: str):
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error removing file: {e}")


@app.post("/fine-tune-and-generate")
async def main(
    background_tasks: BackgroundTasks,
    dataset: UploadFile = File(...),
    model_type: str = Form(...),
    steps: int = Form(3000),
    n_sample: int = Form(1),

):
    if model_type != MODEL_TYPE:
        raise HTTPException(400, f"Model type {model_type} not supported by this container")
    
    # Process dataset
    tmp_dir = tempfile.TemporaryDirectory()
    # Save uploaded file
    dataset_path = f"{tmp_dir}/dataset.zip"
    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    
    # Extract dataset
    extracted_dir = f"{tmp_dir}/extracted"
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)
    
    # Train model
    print("training model")
    train(dataset_path=f"{tmp_dir}/extracted",
            steps=steps,
            outdir=f"{tmp_dir}/model")
    
    # Generate data
    inference(model=f"{tmp_dir}/model", 
                folder_save=f"{tmp_dir}/output",
                num_sample=n_sample)
    
    
    # Zip results
    output_images = glob.glob(f"{tmp_dir}/output/*.png")
    output_zip = f"{tmp_dir}/output.zip"
    with zipfile.ZipFile(output_zip, "w") as zipf:
        for idx, img_path in enumerate(output_images):
            zipf.write(img_path, arcname=f"result_{idx}.png")
    # remove the tmp_dir after send 
    background_tasks.add_task(remove_file, tmp_dir)
    return FileResponse(output_zip, filename="syntheticdata.zip")