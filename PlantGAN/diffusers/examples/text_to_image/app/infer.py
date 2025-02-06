import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline
import random
import gc
import argparse
import os

BATCH_SIZE = 4
NUM_ITER = 212
SAVE_FOLDER = "/home/mia/Downloads/GitHub/PlantGAN/results/gen_stable_diffusion_ckpt2500_mseloss"

def inference(model, folder_save=SAVE_FOLDER, n_infer_steps=25, atten_scale=0.8, guidance_scale=7.5, num_sample=10):
    # load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, 
        use_safetensors=True, safety_checker=None
    )
    pipeline.load_lora_weights(model, weight_name="pytorch_lora_weights.safetensors", adapter_name="plant")

    # load to distributed state
    pipeline.to("cuda")

    prompts = [
        'bell pepper bacterial leaf spot with small dark brown water-soaked lesions',
        'a healthy bell pepper leaf',
        'potato early leaf blight with small dark brown spots that have concentric rings',
        'potato late leaf blight featuring irregular dark spots and lighter green halos',
        'a healthy potato leaf with a lush green appearance and no visible signs of disease',
        'tomato leaf showing symptoms of bacterial leaf spot with small dark brown lesions that have yellow halos',
        'tomato leaf showing symptoms of early blight with dark brown spots that have concentric rings and surrounding yellow halos',
        'tomato late leaf blight with large irregularly shaped dark brown to black lesions pale green or yellowish tissue',
        'tomato leaf mold with yellow blotches and greyish-brown mould',
        'tomato leaf septoria spot with numerous small circular dark brown lesions that have grayish centers and surrounded by yellow halos',
        'tomato leaf two-spot spider mites with visible stippling tiny yellow or white specks and a mottled appearance due to mite feeding',
        'tomato leaf Target Spot with small circular to oval dark brown to black spots',
        'tomato Yellow Leaf Curl Virus with pronounced leaf curling crinkling and a yellowing of the leaf edges',
        'tomato leaf Mosaic Virus with mottled patterns of light and dark green uneven leaf coloring and a general mosaic-like appearance',
        'healthy tomato leaf with a vibrant green color smooth texture showing strong well-defined veins and overall vitalit',
    ]
    prompts = ['detailed, real ' + prompt for prompt in prompts]
    
    os.makedirs(folder_save, exist_ok=True)
    results = []
    for ii in range(0, num_sample, BATCH_SIZE):
        num_images_per_prompt = min(BATCH_SIZE, num_sample - ii)
        # for prompt_id, prompt_ii in enumerate(prompts):
            
        # seed = index_process*NUM_ITER + ii
        prompt_ii = random.choices(prompts, k=num_images_per_prompt)
    # print(new_prompt)
        result = pipeline(prompt_ii, num_inference_steps=n_infer_steps, cross_attention_kwargs={"scale": atten_scale}, 
                            guidance_scale=guidance_scale, height=256, width=256, num_images_per_prompt=num_images_per_prompt, 
                            negative_prompt=['anime, do not have background'] * num_images_per_prompt).images
        for j in range(len(result)):
            result[j].save(f"{folder_save}/result_{ii + j}.png")
        # results.extend(result)

        del result
        torch.cuda.empty_cache()
        gc.collect()
    # return results

def remove_black_image(folder):
    for img in os.listdir(folder):
        path_img = os.path.join(folder, img)
        if os.path.getsize(path_img) <= 842:
            os.remove(path_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--folder_save", type=str, default=SAVE_FOLDER, help='Path to save folder')
    parser.add_argument("--model", type=str, default="plantVillage_text2image_/checkpoint-2500", help="path2lora model")
    parser.add_argument("--n_infer_steps", type=int, default=25, help="number of inference step")
    parser.add_argument('--atten_scale', type=float, default=0.8, help="attention scale")
    parser.add_argument('--guidance_scale', type=float, default=7.5, help="guidance scale")
    
    args = parser.parse_args()

    inference(model=args.model, folder_save=args.folder_save, n_infer_steps=args.n_infer_steps, 
              atten_scale=args.atten_scale, guidance_scale=args.guidance_scale)
    
    remove_black_image(args.folder_save)