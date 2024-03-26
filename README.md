# Cocktail Party Attack

Implementation of the ICML 2023 paper [Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning Using Independent Component Analysis](https://proceedings.mlr.press/v202/kariyappa23a.html).


# Usage
The evaluation pipeline involves three steps as described below:
1. Create venv, download datasets
2. Train a model (or use a pre-trained model e.g. ImageNet)
3. Collect the gradients/model updates (on unseen data)
4. Perform gradient/update inversion using CPA (our proposal) or GMA (prior work)

The commands to run the above steps are as follows:

## 1. Create venv, download datasets

**venv**
conda create --name cpa python=3.8
conda activate cpa
pip install -r requirements.txt

**Tiny-ImageNet**
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P ./datasets/
unzip ./datasets/tiny-imagenet-200.zip -d ./datasets/

**ImageNet**
Download dataset from https://www.image-net.org/download.php and extract it to ./datasets

## 2. Train the model

**Tiny-ImageNet:** python src/train.py --model=fc2 --ds=tiny_imagenet --epochs=20

**ImageNet:** Use Pre-trained model

## 3. Collect the gradients (FedSGD)

**Tiny-Imagenet:** 
python src/collect_grads.py --ds=tiny_imagenet --model=fc2

**ImageNet:**
python src/collect_grads.py --ds=imagenet --model=vgg16

## 4. Perform Gradient Inversion

### CPA

**Tiny-ImageNet:** python src/attack.py --n_log=5000 --model=fc2 --attack=cp --batch_size=128 --decor=1.47 --ds=tiny_imagenet --lr=0.001 --tv=3.1 --T=12.4 --fl_alg=fedsgd

**ImageNet:** python src/attack.py --attack=cp --batch_size=256 --nv=0.13  --l1=5 --T=7.7 --decor=5.3 --fi=1 --tv=0.1 --fi_method=direct --n_sample_fi=16 --ds=imagenet --model=vgg16

### GMA 

**Tiny-ImageNet:** python src/attack.py --n_log=5000 --model=fc2 --ds=tiny_imagenet --attack=gm --batch_size=128 --lr=0.1 --sch=cosine --tv=0.001 --fl_alg=fedsgd

**ImageNet:** python src/attack.py --n_log=5000 --model=vgg16 --ds=imagenet --attack=gm --batch_size=256 --lr=0.1 --sch=cosine --tv=0.001 --fl_alg=fedsgd 

**Note:** For all the hyperparameters, we sweep their values in the range [0.00001, 10] using a single batch of inputs and pick the set of values that yield the best LPIPS score to carry out our attack

# Code Acknowledgements

The majority of Cocktail Party Attack is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [GradAttack](https://github.com/Princeton-SysML/GradAttack/tree/master) is licensed under the MIT license.