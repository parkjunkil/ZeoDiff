# ZeoDiff

![TOC_대지 1](https://github.com/parkjunkil/ZeoDiff/assets/88761984/55831179-9b07-456c-ae6f-0692a7ad964c)

This package provides a diffusion model for the generation of pure silica zeolite, ZeoDiff (Zeolite Diffusion), which generated porous materials using a diffusion model for the first time. The model was developed based on the framework of Denoising Diffusion Probabilistic Model (DDPM) with zeolite structures represented as three dimensional grids of energy, silicon, and oxygen channels. Our model successfully generated realistic zeolite structuers while exhibiting a capability of inverse design with user-desired properties.


---

## Install
    $ git clone git@github.com/parkjunkil/ZeoDiff
    $ pip install -e .

## Training/Test Data Download
    Training and test data are available at "https://doi.org/10.6084/m9.figshare.xxxxxxxxx"
    If you want train the model, please download training.tar.gz and test.tar.gz from the above link, unzip, and locate it under the repository
    
    $ tar -zxvf trianing.tar.gz
    $ tar -zxvf test.tar.gz
    
## Generate New Samples using pre-Trained Model
    
    Following three pre-trianed models are provided within models folder: unconditional.ckpt, conditional_VF.ckpt, and conditional_HOA.ckpt
        unconditional.ckpt : trianed ZeoDiff model without user desirability
        conditional_VF.ckpt : trained ZeoDiff model with conditioned on void fraction
        conditional_HOA.ckpt : trained ZeoDiff model with conditioned on heat of adsorption

    - unconditional 
    $ python run.py with train=True n_samples=1000 model='/models/unconditional.ckpt'

    - conditional (void fraction of 0.20)
    $ python run.py with train=True n_samples=1000 model='/models/conditional_VF.ckpt' target_prop=0.20
    
    - conditional (heat of adsorption of 20 kJ/mol)
    $ python run.py with train=True n_samples=1000 model='/models/conditional_VF.ckpt' target_prop=0.20

## Train New model
    
