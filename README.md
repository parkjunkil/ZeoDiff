# ZeoDiff

![TOC_대지 1](https://github.com/parkjunkil/ZeoDiff/assets/88761984/55831179-9b07-456c-ae6f-0692a7ad964c)

This package provides a diffusion model for the generation of pure silica zeolite, ZeoDiff (Zeolite Diffusion), which generated porous materials using a diffusion model for the first time. The model was developed based on the framework of Denoising Diffusion Probabilistic Model (DDPM) with zeolite structures represented as three dimensional grids of energy, silicon, and oxygen channels. Our model successfully generated realistic zeolite structuers while exhibiting a capability of inverse design with user-desired properties.
[To be linked](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c01822) 

---

## 1. Install
    
    $ git clone git@github.com/parkjunkil/ZeoDiff
    $ pip install setuptools
    $ pip install -e .

---------------------------------------

## 2. Generate New Samples using a pre-Trained Model

### 2.1 pre-trained Models

    Following three pre-trained models/ are provided within models folder:
    
    - unconditional.ckpt : trained ZeoDiff model without user desirability
    - conditional_VF.ckpt : trained ZeoDiff model conditioned on void fraction
    - conditional_HOA.ckpt : trained ZeoDiff model conditioned on heat of adsorption

### 2.2 Examples

#### 2.2.1 unconditional
    
    $ mkdir sample_uncond
    $ python run.py with train=False n_sample=1000 eval_model='unconditional.ckpt' sample_dir='sample_uncond'

#### 2.2.2 conditional (void fraction of 0.20)
    
    $ mkdir sample_vf_0.20
    $ python run.py with train=False self_condition=True target_prop='VF' target_value=0.20 n_sample=1000 eval_model='conditional_VF.ckpt' sample_dir='sample_vf_0.20'
    
#### 2.2.3 conditional (heat of adsorption of 25 kJ/mol)
    
     $ mkdir sample_hoa_25
    $ python run.py with train=False self_condition=True target_prop='HOA' target_value=0.25 n_sample=1000 eval_model='conditional_HOA.ckpt' sample_dir='sample_hoa_25'

---------------------------------------


## 3. Train New model
    
### 3.1 Download Data    

[To be linked](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c01822).

    Training and test data are available at above link.
    If you want to train the model, please download training.tar.gz and test.tar.gz, unzip, and locate it under the repository.
    
    $ tar -zxvf training.tar.gz
    $ tar -zxvf test.tar.gz

    Specify the locations of the training and test directories while running the run.py. Check following examples.

### 3.2 Examples

#### 3.2.1 unconditional
    
    $ python run.py with train=True n_sample=1000 train_dataset='../data/training/' test_dataset='../data/test/'
    
#### 3.2.2 conditional (void fraction)
    
    $ python run.py with train=False n_sample=1000 self_condition=True target_prop='VF' train_dataset='../data/training/' test_dataset='../data/test/'
    
---------------------------------------
    
## 4. Citation

Please consider citing the following paper if you find this package useful.

[To be linked](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c01822) 

---------------------------------------

## 5. Acknowledgements

[Baekjun Kim](https://github.com/good4488) attributed for the data prepartion process and provided useful discussion.

The DDPM codebase and many utility functions are adapted from the [hugging face](https://huggingface.co/blog/annotated-diffusion)

