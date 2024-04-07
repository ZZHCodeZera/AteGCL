# AteGCL
### 1. Note on datasets and directories

Due to the large size of datasets and their similarity matrix *ML-10M*, *Amazon* and *Tmall*, we have uploaded them on BaiduDisk. Please check the URL below.
https://pan.baidu.com/s/1PaDPQHdyVwyCAepKwH7iLg f04m

### 2. Running environment

```
Python version 3.9.12
torch==1.12.0+cu113
numpy==1.21.5
tqdm==4.64.0
```

### 3. How to run the codes

* Yelp

```
python main.py --data yelp
```

* Gowalla

```
python main.py --data gowalla --lambda2 0
```

* ML-10M

```
python main.py --data ml10m --temp 0.5
```

* Tmall

```
python main.py --data tmall --gnn_layer 1
```

* Amazon

```
python main.py --data amazon --gnn_layer 1 --lambda2 0 --temp 0.1
```

### 4. Some configurable arguments

* `--cuda` specifies which GPU to run on if there are more than one.
* `--data` selects the dataset to use.
* `--lambda1` specifies $\lambda_1$, the regularization weight for CL loss.
* `--lambda2` is $\lambda_2$, the L2 regularization weight.
* `--temp` specifies $\tau$, the temperature in CL loss.
* `--dropout` is the edge dropout rate.
* `--q` decides the rank q for PCA.
