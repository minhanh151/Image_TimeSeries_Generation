# APIs for Time Series Generation


## APIs
### TimeGAN
#### How to run
```bash
docker build -f Dockerfile_timegan -t timegan
docker run -p 8004:80 --gpus all timegan
```

#### APIs description
Parameters:

- model_type
- dataset
- data_name
- seq_len
- module
- hidden_dim
- num_layer
- iteration
- batch_size
- n_samples

#### Example

```
cd data
curl --output test.zip -X POST "http://localhost:8004/fine-tune-and-generate"      -F "dataset=@timeseries_ds.zip"      -F "model_type=timegan" -F "iteration=1"
```


### RTSGAN

#### How to run
```bash
docker build -f Dockerfile_rtsgan . -t rtsgan 
docker run -p 8005:80 --gpus all rtsgan
```

#### API description


#### Example
```
cd data
curl --verbose --output test.zip -X POST "http://localhost:8003/fine-tune-and-generate"    -F "dataset=@timeseries_ds.zip"      -F "model_type=rtsgan" -F "epochs=1" -F"iterations=1"
```

### DoppelGAN

#### How to run
```bash
docker build -f Dockerfile_doppelgan . -t doppelgan 
docker run -p 8005:80 --gpus all doppelgan
```

#### API description


#### Example
```
cd data
curl --verbose --output test.zip -X POST "http://localhost:8004/fine-tune-and-generate"      -F "dataset=@timeseries_ds.zip"      -F "model_type=doppelgan" -F "epochs=1"
```

### TTSGAN
#### How to run
```bash
docker build -f Dockerfile_ttsgan . -t ttsgan 
docker run -p 8005:80 --gpus all ttsgan
```

#### API description


#### Example
```
cd data
curl --verbose --output test.zip -X POST "http://localhost:8005/fine-tune-and-generate"      -F "dataset=@timeseries_ds.zip"      -F "model_type=ttsgan" -F "epochs=1" -f "iterations=1" 
```
