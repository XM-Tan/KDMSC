# KDMSC
## This repo is built for the paper: A Lightweight Framework with Knowledge Distillation for Zero-Shot Mars Scene Classification 
### [<a href="https://ieeexplore.ieee.org/document/10699382">Paper</a>]

# Abstract
Gathering extensive labeled data during Mars missions is costly and unrealistic, especially considering the complex and unpredictable Martian environment where new and unfamiliar scenes may emerge. Traditional Mars scene classification (MSC) methods depend heavily on large amounts of labeled data, which makes it impractical to recognize previously unseen scene classes without the necessary labeled examples. In addition, the significant computational demands and parameter requirements of modern models also pose challenges for their integration into resource-constrained systems used in Mars exploration. To address these issues, we propose a zero-shot MSC (ZSMSC) framework, which is able to categorize unseen Martian image scenes without the prior acquisition of vast visual examples. Specifically, the framework combines lightweight model design with knowledge distillation (KD) techniques, known as KDMSC, to streamline complex zero-shot learning (ZSL) models. It employs a KD loss that captures essential knowledge through the training of the teacher model from scratch, thereby improving the zero-shot classification performance of the student model. Consequently, the lightweight student model is tailored for deployment on devices with limited resources while fulfilling the requirements of the ZSMSC tasks. Moreover, to support the ZSMSC initiative, we developed a dataset named ZSMars to further advance this field. Experimental results indicate that our model excels in the ZSMSC tasks while maintaining low computational complexity and storage requirements.

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
