# Self-Supervised Learning Models

A PyTorch implementation of popular self-supervised models

### Installation

```
git clone https://github.com/youshyee/SSL.git
cd SSL
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Self-Supervised training

Change the workdir path in configs, and data path
Then start training
```
sh dist_train.sh $GPU_YOU_HAVE $CONFIG_PATH
```

### Run SimSiam

```
sh dist_train.sh $GPU_NUM ./configs/simsiam.py
```

### Run SimCLR

```
sh dist_train.sh $GPU_NUM ./configs/SimCLR.py
```

### Runing on Slurm
- Change the training file location
- Change the PYTHON path file
- Change the Config file path
- Change the $GPU to gpu num you want to use

Then run

```
sbatch st_train.sh $GPU_NUM ./configs/SimCLR.py
```
### TODO

More model BYOL MOCO SimCLRv2 etc  **Stay Tuned**





