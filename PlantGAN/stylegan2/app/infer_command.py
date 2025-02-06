import sys
# sys.path.append('stylegan2')
# sys.path.append('stylegan2/dnnlib')
import torch
import random
import gc
import argparse
import dnnlib
import numpy as np 
# import stylegan2.dnnlib as dnnlib
import os
import legacy
from PIL import Image
# import stylegan2.legacy as legacy

BATCH_SIZE = 4
NUM_ITER = 212
SAVE_FOLDER = "/home/mia/Downloads/GitHub/PlantGAN/results/gen_stable_diffusion_ckpt2500_mseloss"

def load_gan(path='/workspace/PlantGAN/stylegan2/plantvillage/00005-PlantVillage-cond-auto4/network-snapshot-023788.pkl'):
    #load all the model
    with dnnlib.util.open_url(path) as f:
        stylegan = legacy.load_network_pkl(f)['G_ema'].to('cuda') # type: ignore
    return stylegan

def inference(model, folder_save=SAVE_FOLDER, num_sample=10):
    G = load_gan(model)
    results = []
    os.makedirs(folder_save, exist_ok=True)
    for ii in range(0, num_sample, BATCH_SIZE):
        num_images_per_iter = min(BATCH_SIZE, num_sample - ii)

        label = torch.zeros([num_images_per_iter, G.c_dim], device='cuda')
        classes = random.choices(range(G.c_dim), k=num_images_per_iter)
        label[torch.arange(num_images_per_iter), torch.tensor(classes)] = 1

        z = torch.from_numpy(np.random.RandomState( + ii).randn(num_images_per_iter, G.z_dim)).to('cuda')
        img = G(z, label, truncation_psi=1, noise_mode='const')
        img_show = (img.clone().permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img_show = np.array(img_show.cpu())
        img_show = [Image.fromarray(im) for im in img_show]
    # print(new_prompt)

        for j in range(len(img_show)):
            img_show[j].save(f"{folder_save}/result_{ii + j}.png")
        del img_show
        # results.extend(img_show)
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

    
    args = parser.parse_args()

    inference(model=args.model, folder_save=args.folder_save, num_sample=10)
    
    remove_black_image(args.folder_save)