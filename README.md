<div align="center">
<!-- <h1> MoDA </h1> -->
<h2> <a href="">MoDA: Unlocking Frozen Vision Foundation Models for Multimodal Fusion Object Detection in Remote Sensing</h2>
<h3> Chao Wang, Yanguang Sun, Zhenbo Yu, Jian Yang, Lei Luo*</h3>
</div>

#### This repository contains the official implementation of the paper: MoDA: Unlocking Frozen Vision Foundation Models for Multimodal Fusion Object Detection in Remote Sensing

## ✨ Overview

<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/overall-network.png" width="99.5%"> </p>



## 📄 Documentation
### Installation
Create and activate a conda environment:
```
conda create -n moda python=3.11
conda activate moda
```
Install the required packages:
```
git clone https://github.com/wchao0601/MoDA.git
cd MoDA/
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
pip install -r requirements.txt
```

### Train
```python
python train.py
```

### Test
```python
python test.py
```


## 📈 Results
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/results1.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/results2.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/vis-detection.png" width="99.5%"> </p>
<p align="center"> <img src="https://github.com/wchao0601/MSOD-DGIoU/blob/main/vis-heatmap.png" width="99.5%"> </p>

## 🌐 Contact
If you have any questions, please feel free to contact me via email at wchao0601@163.com

## 📚 Citation
If our work is helpful, you can cite our paper:
```

```
## 🙏 Acknowledgment
- This repo is based on [Ultralytics](https://github.com/ultralytics/ultralytics), which is excellent works.
