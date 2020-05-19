# Class-aware and Adversarial Latent Feature Generation for Imbalanced Classification

> Class-aware and Adversarial Latent Feature Generation for Imbalanced Classification
>
> 

This is the official implementation of *CaLeG*.
## Abstract
In real-world applications, discovering hidden pattern from imbalanced data
is a challenging issue. Due to the limitation of data especially for minority
classes, existing classification methods usually suffer from unstable prediction
and low performance. In this paper, a deep Class-aware and adversarial Latent
feature Generation (CaLeG) is proposed for imbalanced classification. Its main
goal is to sufficiently determine the Direct Causes of the target labels even
when there are few samples. Specially, CaLeG mines the latent features from 
the original data via a supervised variation autoencoder model. Then, CaLeG 
augments the latent features for each minority classes by constructing 
class-aware Gaussian distribution of the learnt latent features. To guarantee 
both the learnt and augmented latent features generate realistic data and have 
discriminative ability, an adversarial network is introduced to adjust the 
direct cause learning process. Extensive experiments have been conducted on 
widely-used real imbalanced image datasets. By comparing with the popular 
imbalanced classification baselines, the experimental results demonstrate the 
superiority of our proposed model in terms of several evaluation metrics

## Requirement

The code was tested on:
- python=3.7
- tensorflow=1.15.0
- torchvision=0.5.0 (utilizd for dataset preparation)


## Usage
```
python run.py -h
usage: run.py [-h] [--imbalance IMBALANCE] [--aug-rate AUG_RATE]
              [--dataset {mnist,fashion,cifar10,svhn}]

optional arguments:
  -h, --help            show this help message and exit
  --imbalance IMBALANCE
                        whether to use imbalanced data
  --aug-rate AUG_RATE   sampling rate r
  --dataset {mnist,fashion,cifar10,svhn}
                        dataset name

```
The dataset will be automatically downloaded and prepared in `./data` when first run.

## License
MIT