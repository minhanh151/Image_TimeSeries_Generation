import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import zipfile
import shutil
from run_command import train
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
    model_type: str = Form(...),
    dataset: UploadFile = File(...),
    data_name: str = Form('stock'),
    sample_len: int = Form(6),
    seq_len: int = Form(24),
    epochs: int = Form(50000),
    batch_size: int = Form(128),
    n_samples: int = Form(1),
):
    if model_type != MODEL_TYPE:
        raise HTTPException(400, f"Model type {model_type} not supported by this container")
    
    # Process dataset
    tmp_dir = tempfile.mkdtemp()
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
          data_name=data_name,
          seq_len=seq_len,
          epochs=epochs,
          batch_size=batch_size,
          n_samples=n_samples,
          sample_len=sample_len,
          outdir=tmp_dir)

    
 
    # output_images = glob.glob(f"{tmp_dir}/output/*.png")
    output_zip = f"{tmp_dir}/output.zip"
    with zipfile.ZipFile(output_zip, "w") as zipf:
        # for idx, img_path in enumerate(output_images):
        zipf.write(f'{tmp_dir}/data.npy', arcname=f"sync_data.npy")
    # remove the tmp_dir after send 
    background_tasks.add_task(remove_file, tmp_dir)
    return FileResponse(output_zip, filename="syntheticdata.zip")