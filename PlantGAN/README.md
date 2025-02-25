# APIs for Image Generation


## APIs
### Diffuser
#### How to run
```bash
cd diffusers/examples/text_to_image
docker build -t diffusers
docker run -p 8000:80 --gpus all diffusers
```

#### APIs description




#### Example

```
curl -X POST "http://localhost:8000/fine-tune-and-generate"   -F "dataset=@path_to_dataset"  -F "model_type=sd-lora" -F "steps=1"
```
### StyleGan2

#### How to run
```bash
cd stylegan2
docker build . -t stylegan2 
docker run -p 8001:80 --gpus all stylegan2
```

#### API description


#### Example
```
curl -X POST "http://localhost:8001/fine-tune-and-generate"      -F "dataset=@path_to_dataset.zip"      -F "model_type=stylegan2"
```
