# KDMSC
## This repo is built for the paper: A Lightweight Framework with Knowledge Distillation for Zero-Shot Mars Scene Classification 
### [<a href="https://ieeexplore.ieee.org/document/10699382">Paper</a>]

# Getting Started
## Installation
### Step 1: Clone the KDMSC repository:
To get started, first clone the KDMSC repository and navigate to the project directory:
```
git clone https://github.com/XM-Tan/KDMSC.git
cd KDMSC
```
### Step 2: Environment Setup:
MFINet recommends setting up a conda environment and installing dependencies via pip. 
Use the following commands to set up your environment:

***Creat and activate a new conda environment***

```
conda create -n KDMSC python=3.9
conda activate KDMSC
```

***Install Dependencies***

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

*Install Transformer*

```
pip install huggingface_hub
conda install -c huggingface transformers==4.16.2
```

*Others*

```
pip install scikit-learn
pip install timm
pip install h5py
```

# Model Training and Testing

To train and test KDMSC for zero-shot classification on ZSMars, use the following commands for different configurations:

```
python ./main_mars_dis.py --dataset ZSMars --batch_size 7 --manualSeed 42 --seen_unseen_ratio 64
```
# Citation
If it is helpful for your work, please cite this paper:
``` 
@ARTICLE{10699382,
  author={Tan, Xiaomeng and Xi, Bobo and Xu, Haitao and Li, Jiaojiao and Li, Yunsong and Xue, Changbin and Chanussot, Jocelyn},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Lightweight Framework with Knowledge Distillation for Zero-Shot Mars Scene Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Mars;Visualization;Semantics;Feature extraction;Scene classification;Image recognition;Microwave integrated circuits;Accuracy;Transformers;Data models;Mars scene classification;zero-shot learning;knowledge distillation;lightweight model},
  doi={10.1109/TGRS.2024.3470526}}

```
