# GMC

##### This repository contains code for the paper *Fair Risk Control: A Generalized Framework for Calibrating Multi-group Fairness Risks*.

### Sturcture

The `src/` folder contains Python implementations for our GMC algorithm

The `experiments/` folder contains Jupyter notebooks implementing the experiments in the paper.

The `datasets/` folder contains Python implementations to preprocess the data.

The `fig/` folder contains the result figures as shown in the paper.

### Requirements

**The required packages for *FigSeg* is:**

* python>=3.8

* pytorch>=1.7.1

* scipy>=1.10.1

* numpy>=1.24.4

**The required packages for *HierClass* is:**

* python>=3.8

* pytorch>=2.0.1

* datasets

* torch-geometric == 1.7.2

* torch-scatter == 2.0.8

* torch-sparse == 0.6.12

**The required packages for *FairText* is:**

* python>=3.8

* transformers ==4.32.1

* tokenizers==0.13.2

* pytorch>=2.0.1

### Preprocess

For three simulations, we all need to preprocess the data to get the initial scoring functions.

##### FigSeg

The original dataset can be acquired in [FASSEG-repository](https://github.com/massimomauro/FASSEG-repository). The U-net model implementation could refer to https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/unet/__init__.py and we provide a copy here.

To generate the scoring functions, run `FigSeg.py` in `Datasets/FigSeg`

##### HierClass

The original dataset can be acquired in [the repository of HDLTex](https://github.com/kk7nc/HDLTex). Preprocessing codes could refer to [the repository of HiAGM](https://github.com/Alibaba-NLP/HiAGM) and we provide a copy here. Instead of repeating the training, we use the checkpoints of the original repository and yield the results directly.

To generate the scoring functions, use

```shell
cd Datsets/HierClass/data/WebOfScience
python preprocess_wos.py
python data_wos.py
python test.py --name WebOfScience-HPT
```

##### FairText

To generate the scoring functions, run `FairText.py` in `Datasets/FairText`

### 

### Main Experiments

After getting the initial score functions in the preprocessing stage, we run the main experiments and get the figures shown in the paper. We provide a copy of the figures in `fig/`

##### FigSeg

Run `FairFig.ipynb` in `experiments/`

##### HierClass

Run `HierClass.ipynb` in `experiments/`

##### FairText

Run `FairText.ipynb` in `experiments/`

### Reproducibility

Our experiment is implemented on GeForce MX250 GPU with CUDA version 10.1. The random seeds for *FigSeg*, *HierClass* and *FairText* are 42, 1 and 16 respectively.
