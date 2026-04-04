# Medical 2D Image Segmentation Benchmarks
##  1. Dataset Preparation

The dataset is hosted on Zenodo.


Please download the dataset from Zenodo:

```
https://doi.org/10.5281/zenodo.19201948
```
Please put the dataset as the following architecture. 

```
├── segmentation
    ├── data
        ├── images
        ├── masks
    ├── src
    ├── main.py
    ├── split.py
```

## 2.Environments

- GPU: NVIDIA GeForce RTX4090 GPU
- Pytorch: 1.13.0 cuda 11.7
- cudatoolkit: 11.7.1
- scikit-learn: 1.0.2
- albumentations: 1.2.0

## 3.Training

You can train and validate dataset:

```python
python main.py --model [CMUNeXt/CMUNet/TransUnet/...] --base_dir ./data/busi --train_file_dir busi_train.txt --val_file_dir busi_val.txt --base_lr 0.01 --epoch 300 --batch_size 8
```

## 4.Inference

```python
python infer.py --model [CMUNeXt/CMUNet/TransUnet/...] --model_path [.pth] --base_dir ./data/busi --val_file_dir busi_val.txt --img_size 256 --num_classes 1
```
## 5.Semantic Segmentation Code

The semantic segmentation part of this project is adapted from a GitHub open-source project, but the original repository is no longer accessible.  
The local code and dependencies are preserved, allowing replication of the related experiments and results.
