import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import tempfile
import zipfile
import shutil
from train import train
from infer import inference
# from infer import generate

app = FastAPI()
MODEL_TYPE = os.getenv("MODEL_TYPE", "default")

@app.post("/fine-tune-and-generate")
async def main(
    dataset: UploadFile = File(...),
    model_type: str = Form(...),
    n_infer_steps: int = Form(10),
    steps: int = Form(3000),
    atten_scale: float = Form(0.8),
    guidance_scale: float = Form(7.5),

):
    if model_type != MODEL_TYPE:
        raise HTTPException(400, f"Model type {model_type} not supported by this container")
    
    # Process dataset
    with tempfile.TemporaryDirectory() as tmp_dir:
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
        print("generating data")
        inference(model=f"{tmp_dir}/model", 
                  folder_save=f"{tmp_dir}/output", 
                  n_infer_steps=n_infer_steps, 
                  atten_scale=atten_scale, 
                  guidance_scale=guidance_scale,
                  num_sample=1)
        
        
        # Zip results
        output_zip = f"{tmp_dir}/output.zip"
        with zipfile.ZipFile(output_zip, "w") as zipf:
            for idx, img_path in enumerate(output_images):
                zipf.write(img_path, arcname=f"result_{idx}.png")

        return FileResponse(output_zip, filename="syntheticdata.zip")