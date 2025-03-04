# APIs for Image and Time Series Generation 

## Introductory
This repository provides different APIs using FastAPI with generatives models. 


## Dataset
The format for image generation dataset is:

The format for time-series generation dataset is:


The sample dataset are at the `data` folder. 
+ `dataset.zip` for image-genration dataset
+ `test.zip` for time-series generation dataset 

## APIs
If you want to try out only one API at one time. The following APIs are currently available:

1. [Image Generation](PlantGAN/README.md)
    + Stable-Diffusion
    + StyleGAN2
    + CGAN
    + ControlNet #todo
2. [Time-Series Generation](Time-Gen-AI/README.md)
    + TimeGAN
    + DoppelGAN 
    + RTSGAN
    + TTS-GAN

## Build a server
If you want to run all the models at the same time. 

Run file docker-compose
```
docker-compose up --build 
``` 