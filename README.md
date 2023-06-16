# ZeoDiff

![TOC_대지 1](https://github.com/parkjunkil/ZeoDiff/assets/88761984/55831179-9b07-456c-ae6f-0692a7ad964c)

This package provides a diffusion model for the generation of pure silica zeolite, ZeoDiff (Zeolite Diffusion), which generated porous materials using a diffusion model for the first time. The model was developed based on the framework of Denoising Diffusion Probabilistic Model (DDPM) with zeolite structures represented as three dimensional grids of energy, silicon, and oxygen channels. Our model successfully generated realistic zeolite structuers while exhibiting a capability of inverse design with user-desired properties.
[To be linked](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c01822) 

---

## Install
    
    $ git clone git@github.com/parkjunkil/ZeoDiff
    $ pip install -e .

## Training/Test Data Download
    
Training and test data are available at [To be linked](https://pubs.acs.org/doi/full/10.1021/acs.chemmater.2c01822).
If you want to train the model, please download training.tar.gz and test.tar.gz from the above link, unzip, and locate it under the repository.
    
    $ tar -zxvf training.tar.gz
    $ tar -zxvf test.tar.gz
    
## Generate New Samples using pre-Trained Model
    
    Following three pre-trained models are provided within models folder:
    
    - unconditional.ckpt : trained ZeoDiff model without user desirability
    - conditional_VF.ckpt : trained ZeoDiff model conditioned on void fraction
    - conditional_HOA.ckpt : trained ZeoDiff model conditioned on heat of adsorption

unconditional
    
    $ python run.py with train=False n_sample=1000 model='/models/unconditional.ckpt'

conditional (void fraction of 0.20)
    
    $ python run.py with train=False self_condition=True target_prop='VF' target_value=0.20 n_sample=1000 model='/models/conditional_VF.ckpt' sample_dir='sample_vf_0.20'
    
onditional (heat of adsorption of 25 kJ/mol)
    
    $ python run.py with train=False self_condition=True target_prop='HOA' target_value=0.25 n_sample=1000 model='/models/conditional_HOA.ckpt' sample_dir='sample_hoa_25'

## Train New model
    
