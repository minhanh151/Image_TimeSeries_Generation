# APIs for Time Series Generation


## APIs
### TimeGAN
#### How to run
```bash
docker build -f Dockerfile_timegan -t timegan
docker run -p 8004:80 --gpus all timegan
```

#### APIs description




#### Example

```
curl --output test.zip -X POST "http://localhost:8004/fine-tune-and-generate"      -F "dataset=@dataset.zip"      -F "model_type=timegan" -F "iteration=1"
```


### RTSGAN

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
