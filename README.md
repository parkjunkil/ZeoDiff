# ZeoDiff

![TOC_대지 1](https://github.com/parkjunkil/ZeoDiff/assets/88761984/55831179-9b07-456c-ae6f-0692a7ad964c)

This package provides a diffusion model for the generation of pure silica zeolite, ZeoDiff (short for Zeolite Diffusion), which generated porous materials using a diffusion model for the first time. The model was developed based on the framework of Denoising Diffusion Probabilistic Model (DDPM) with zeolite structures represented as three dimensional grids of energy, silicon, and oxygen channels. Our model successfully generated realistic zeolite structures while exhibiting a capability of inverse design with user-desired properties.
[Chemrxiv](https://chemrxiv.org/engage/chemrxiv/article-details/64b62833ae3d1a7b0de69d62) 

---

## 1. Install

    We encourage users to build separate anaconda environment with python version >= 3.9.  GPU machine is required for the training.
    
    $ git clone https://github.com/parkjunkil/ZeoDiff.git
    $ conda crate -name zeodiff python=3.9
    $ conda activate zeodiff
    $ pip install -r requirements.txt

---------------------------------------

## 2. Generate New Samples using a pre-Trained Model

### 2.1 pre-trained Models

    Following three pre-trained models are provided within models/ folder:
    
    - unconditional.ckpt : trained ZeoDiff model without user desirability
    - conditional_VF.ckpt : trained ZeoDiff model conditioned on void fraction
    - conditional_HOA.ckpt : trained ZeoDiff model conditioned on heat of adsorption

### 2.2 Examples

#### 2.2.1 unconditional

    $ cd zeodiff
    $ mkdir sample_uncond
    $ python run.py with train=False n_sample=1000 eval_model='unconditional.ckpt' sample_dir='sample_uncond'

#### 2.2.2 conditional (void fraction of 0.20)
    
    $ cd zeodiff
    $ mkdir sample_vf_0.20
    $ python run.py with train=False self_condition=True target_prop='VF' target_value=0.20 n_sample=1000 eval_model='conditional_VF.ckpt' sample_dir='sample_vf_0.20'
    
#### 2.2.3 conditional (heat of adsorption of 25 kJ/mol)

    $ cd zeodiff
    $ mkdir sample_hoa_25
    $ python run.py with train=False self_condition=True target_prop='HOA' target_value=0.25 n_sample=1000 eval_model='conditional_HOA.ckpt' sample_dir='sample_hoa_25'

---------------------------------------


## 3. Train New Model
    
### 3.1 Download Data    

[Figshare](https://figshare.com/articles/dataset/ZeoDiff/23538738)

    Training and test data are available at above link.
    If you want to train the model on your own, please download 'training.tar.gz' and 'test.tar.gz', unzip, and locate it under '/ZeoDiff/data/'.
    
    $ tar -zxvf training.tar.gz
    $ tar -zxvf test.tar.gz

    Specify the locations of the training and test directories while running the run.py. Check following examples.

### 3.2 Examples

    In addition to tags handled in following examples, you also need to change 'n_gpu', 'devices', 'num_nodes' depending on your environment.

#### 3.2.1 unconditional
    
    $ cd zeodiff
    $ python run.py with train=True train_dataset='../data/training/' test_dataset='../data/test/'
    
#### 3.2.2 conditional (void fraction)
    
    $ cd zeodiff
    $ python run.py with train=True self_condition=True target_prop='VF' train_dataset='../data/training/' test_dataset='../data/test/'
    
---------------------------------------
    
## 4. Citation

Please consider citing the following paper if you find this package useful.

[Chemrxiv](https://chemrxiv.org/engage/chemrxiv/article-details/64b62833ae3d1a7b0de69d62) 

---------------------------------------

## 5. Acknowledgements

I greatly appreciate [Baekjun Kim](https://github.com/good4488) for the useful discussion.

The DDPM baseline code and many other utility functions are adapted from the [hugging face](https://huggingface.co/blog/annotated-diffusion)

