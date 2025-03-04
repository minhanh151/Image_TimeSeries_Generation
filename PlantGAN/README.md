# APIs for Image Generation


## Introcution
The following reporatory contains the APIs for generating images regarding latest models:

+ [Diffusers](### Diffuser)

### Diffuser
#### How to run
```bash
cd diffusers/examples/text_to_image
docker build -t diffusers
docker run -p 8000:80 --gpus all diffusers
```

#### APIs description

TODO 



#### Example

```
cd data
curl -X POST "http://localhost:8000/fine-tune-and-generate"   -F "dataset=@stable_diff_ds.zip"  -F "model_type=sd-lora" -F "steps=1"
```
### StyleGan2

#### How to run
```bash
cd stylegan2
docker build . -t stylegan2 
docker run -p 8001:80 --gpus all stylegan2
```

#### API description

TODO

#### Example
```
cd data
curl -X POST "http://localhost:8001/fine-tune-and-generate"      -F "dataset=@stylegan2_ds.zip"      -F "model_type=stylegan2"
```

### CGAN

#### How to run
```bash
cd stylegan2
docker build . -t stylegan2 
docker run -p 8001:80 --gpus all stylegan2
```

#### API description

TODO

#### Example
```
cd data
curl -X POST "http://localhost:8001/fine-tune-and-generate"      -F "dataset=@stylegan2_ds.zip"      -F "model_type=stylegan2"
```