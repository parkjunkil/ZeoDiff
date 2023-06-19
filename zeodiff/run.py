import os
import sys

import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
import copy

from dataset import GridDataModule

from Unet import Unet
from Unet_Reg import Unet_Reg
from diffusion import GaussianDiffusion, linear_beta_schedule

from config import config as _config
from config import ex

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DDPM import DDPM
from Unet_Reg import Unet_Reg

import json


def run(log_dir='logs/', train=True, **kwargs):
	
	config = _config()
	for key in kwargs.keys():
		assert key in config, 'wrong config arguments are given as an input.'

	config.update(kwargs)
	config['log_dir'] = log_dir
	config['train'] = train.lower()=='true'

	main(config)

@ex.automain
def main(_config):
	
	pl.seed_everything(_config["seed"])

	_config = copy.deepcopy(_config)
	
	exp_name = f"{_config['exp_name']}"

	denoising_diffusion_model = DDPM(_config)

	grid_data_module = GridDataModule(_config)
    
	logger = pl.loggers.tensorboard.TensorBoardLogger(
		_config["log_dir"],
		name=f'{exp_name}_seed_{_config["seed"]}_train_{_config["train"]}_batchsize_{_config["batch_size"]}_target_{_config["target_prop"]}_lr_{_config["lr"]}_timesteps_{_config["timesteps"]}_loss_{_config["loss_type"]}',
	)

	# Training
	if _config["train"]:

		if _config["load_model"] is not None:
			load_model_dir = os.path.join(_config['model_dir'], _config['load_model'])
			state_dict = torch.load(load_model_dir)
			denoising_diffusion_model.load_state_dict(state_dict['state_dict'])

		os.makedirs(_config["log_dir"], exist_ok=True)
		checkpoint_callback = ModelCheckpoint(filename=os.path.join(_config['save_dir'], 'ddpm_{epoch:02d}-{val_loss:.6f}'),
												monitor='val_loss',
												verbose=True,
												save_last=True,
												save_top_k=1,
												mode='min',
											)


		early_stopping_callback = EarlyStopping(monitor='val_loss', patience=_config['early_stopping'], verbose=True)


		trainer = pl.Trainer(
							accelerator = _config['accelerator'],
							devices = _config['devices'],
							num_nodes = _config["num_nodes"],
							max_epochs = _config['max_epochs'],
							precision = _config['precision'],
							callbacks = [checkpoint_callback, early_stopping_callback],
							logger = logger,
							strategy = DDPStrategy(find_unused_parameters=True),
							#strategy = _config["strategy"],
							)

		trainer.fit(denoising_diffusion_model, grid_data_module)
	
	# Sampling
	else:

		denoising_diffusion_model.cuda()
		load_model_dir = os.path.join(_config['model_dir'], _config['eval_model'])
		state_dict = torch.load(load_model_dir)
		denoising_diffusion_model.load_state_dict(state_dict['state_dict'])
		denoising_diffusion_model.eval()

		if not os.path.exists(_config['sample_dir']):
			os.mkdir(_config['sample_dir'])
	
		cell_model  = Unet_Reg(
							dim = _config['dim'],
							dim_mults = _config['c_dim_mults'],
							channels = _config['channels'],
							)
		cell_model.load_state_dict(torch.load(_config['c_model_dir']))


		if _config['self_condition'] == False:
			target_value = None

		denoising_diffusion_model.diffusion.large_sample(denoising_diffusion_model.ema,
														cell_model,
														_config['n_sample'],
														_config['sample_dir'],
														target_value = _config['target_value'],
														)


