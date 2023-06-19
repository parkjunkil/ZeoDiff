import torch
import pytorch_lightning as pl
import numpy as np
import os
from torch.utils.data import Dataset, random_split, DataLoader
import matplotlib.pyplot as plt
from config import config as _config
from util import Void_Fraction, Henry_Coeff, Heat_of_Adsorption
import pickle

import copy

# Read energy grid, normalize, and return it as tensor
def read_and_normalize(data, grid_size = 32, upper_scale = 5000, lower_scale = -4000):

		# read griddata file and resize into (32,32,32)
		arr = np.fromfile(data, dtype='float32')
		arr = arr.reshape(grid_size, grid_size, grid_size)

		# truncate outliers and normalize to (-1,1)
		arr = np.where(arr < upper_scale, arr, upper_scale)
		arr = np.where(arr > lower_scale, arr, lower_scale)

		arr = ((arr-lower_scale) / (upper_scale-lower_scale) - 0.5) * 2
		
		# inverse value
		arr = - arr

		return torch.tensor(arr)    

# Read atomic (silicon or oxygen) grid, normalize and return it as tensor
def read_and_normalize_atom(data, grid_size = 32):

	# read griddata file and resize into (32,32,32)
    arr = np.fromfile(data, dtype='float32')
    arr = arr.reshape(grid_size, grid_size, grid_size)

	# normalize to [-1,1]
    arr = (arr-np.min(arr))/(np.max(arr)-np.min(arr))

    arr = (arr - 0.5) * 2

	# inverse value 
    arr = - arr

    return torch.tensor(arr)

    
# For rotational augmentation
def rotate(matrix1, matrix2, matrix3):

	matrix1 = matrix1.reshape(32,32,32)
	matrix2 = matrix2.reshape(32,32,32)
	matrix3 = matrix3.reshape(32,32,32)

	direction = np.random.choice(['x', 'y', 'z'])

	if direction == 'x':
		return torch.tensor(np.rot90(matrix1, k=1, axes=(1, 2)).copy()), torch.tensor(np.rot90(matrix2, k=1, axes=(1, 2)).copy()), torch.tensor(np.rot90(matrix3, k=1, axes=(1, 2)).copy())
	elif direction == 'y':
		return torch.tensor(np.rot90(matrix1, k=1, axes=(0, 2)).copy()), torch.tensor(np.rot90(matrix2, k=1, axes=(0, 2)).copy()), torch.tensor(np.rot90(matrix3, k=1, axes=(0, 2)).copy())
	else:
		return torch.tensor(np.rot90(matrix1, k=1, axes=(0, 1)).copy()), torch.tensor(np.rot90(matrix2, k=1, axes=(0, 1)).copy()), torch.tensor(np.rot90(matrix3, k=1, axes=(0, 1)).copy())

# For translational augmentation
def translate(matrix1, matrix2, matrix3):
    
	dim = matrix1[0].shape[0]

	dx, dy, dz = np.random.randint(-dim//2, dim//2), np.random.randint(-dim//2, dim//2), np.random.randint(-dim//2, dim//2)

	tr_matrix1 = np.roll(matrix1, shift=dx, axis=0)
	tr_matrix1 = np.roll(tr_matrix1, shift=dy, axis=1)
	tr_matrix1 = np.roll(tr_matrix1, shift=dz, axis=2) 

	tr_matrix2 = np.roll(matrix2, shift=dx, axis=0)
	tr_matrix2 = np.roll(tr_matrix2, shift=dy, axis=1)
	tr_matrix2 = np.roll(tr_matrix2, shift=dz, axis=2) 

	tr_matrix3 = np.roll(matrix3, shift=dx, axis=0)
	tr_matrix3 = np.roll(tr_matrix3, shift=dy, axis=1)
	tr_matrix3 = np.roll(tr_matrix3, shift=dz, axis=2) 

	return torch.tensor(tr_matrix1), torch.tensor(tr_matrix2), torch.tensor(tr_matrix3)

# Visualize grids using matplotlib (not necessary)
def visualize_3d(data, grid_idx):
	fig = plt.figure()
    
	data = data[grid_idx,:,:,:]
       
	ax = fig.add_subplot(projection='3d')
 
	x,y,z = np.mgrid[0:1:32j, 0:1:32j, 0:1:32j]
    
	if grid_idx == 0:
		ax.scatter(x, y, z, c=data, cmap='Reds', s=5, alpha=0.1)    
	else:
		ax.scatter(x, y, z, c=-data, cmap='Reds', s=5, alpha=0.5)

# Read three grids and conduct data augmentation
def read_and_normalize_3grid(file_address, augmentation, grid_size=32):

	grid_file = file_address+'.griddata'
	si_file = file_address+'.si'
	o_file = file_address+'.O'
    
	energy_data = read_and_normalize(grid_file, grid_size).reshape(1,grid_size,grid_size,grid_size)
	si_data = read_and_normalize_atom(si_file, grid_size).reshape(1,grid_size,grid_size,grid_size)
	o_data = read_and_normalize_atom(o_file, grid_size).reshape(1,grid_size,grid_size,grid_size)

	original_data = torch.concat((energy_data,si_data,o_data),dim=0)

	rotated_data = None
	translated_data = None

	if augmentation:
		energy_data_rot, si_data_rot, o_data_rot = rotate(energy_data, si_data, o_data)

		energy_data_rot = energy_data_rot.reshape(1,grid_size,grid_size,grid_size)
		si_data_rot =si_data_rot.reshape(1,grid_size,grid_size,grid_size)
		o_data_rot = o_data_rot.reshape(1,grid_size,grid_size,grid_size)

		rotated_data = torch.concat((energy_data_rot,si_data_rot,o_data_rot),dim=0)

		energy_data_trans, si_data_trans, o_data_trans = translate(energy_data, si_data, o_data)     
		translated_data = torch.concat((energy_data_trans, si_data_trans, o_data_trans),dim=0)
        
	return original_data, rotated_data, translated_data

# Prepare Dataset with data augmentation. VF, HC, HOA of each sample are also included
class Three_Grid_Dataset(Dataset):
	def __init__(self, data_dir, property_file, grid_size, augmentation, test = False):
		self.data_dir = data_dir
		self.grid_size = grid_size
		self.property_file = property_file
		all_data_path = os.listdir(data_dir)
		griddata_path = [x for x in all_data_path if x.endswith('griddata')]

		self.augmentation = augmentation

		self.griddata_path_list = griddata_path
		if test:
			self.griddata_path_list = griddata_path[:100]

		with open(self.property_file,'rb') as f:
			prop_dict = pickle.load(f)

		self.prop_dict = prop_dict

		self.max_props = [max(col) for col in zip(*prop_dict.values())]

		len_data = len(self.griddata_path_list)
        
		# only for test trial.
		if test:
			len_data = 100

		if self.augmentation:
			self.len =  len_data * 3
		else:
			self.len =  len_data


	def __len__(self):
		return self.len

	def __getitem__(self, idx):

		if not self.augmentation: 
			data_energy_path = os.path.join(self.data_dir, self.griddata_path_list[idx])
			data_path = data_energy_path[:-9]

			prop_key = data_energy_path.split('/')[-1]

			props = self.prop_dict[prop_key]

           	#normalize
			props[0] = props[0]
			props[1] = props[1] / 500
			props[2] = props[2] / 100

			original_data, _, _ = read_and_normalize_3grid(data_path, self.augmentation)

			return torch.Tensor(original_data), torch.Tensor(props)
        
		else:
			path_idx = idx % len(self.griddata_path_list)

			data_energy_path = os.path.join(self.data_dir, self.griddata_path_list[path_idx])
			data_path = data_energy_path[:-9]

			prop_key = data_energy_path.split('/')[-1]
            
			props = self.prop_dict[prop_key]

			#normalize
			props[0] = props[0]
			props[1] = props[1] / 500
			props[2] = props[2] / 100

			original_data, rotated_data, translated_data = read_and_normalize_3grid(data_path, self.augmentation)

			if idx < len(self.griddata_path_list):
				get_data = original_data

			elif idx < 2 * len(self.griddata_path_list):
				get_data = rotated_data

			else:
				get_data = translated_data


			return torch.Tensor(get_data), torch.Tensor(props)

# Data module used for pytorch lightning
class GridDataModule(pl.LightningDataModule):
	def __init__(self,_config):
		super().__init__()	
		self.config = _config
		self.batch_size = self.config['batch_size']
		self.grid_size = self.config['dim']
		self.train_data_dir = self.config['train_dataset']
		self.test_data_dir = self.config['test_dataset']
		self.test = self.config['test_only_100']
		self.augmentation = self.config['augmentation']
		self.property_file = self.config['property_file']
		self.num_workers = self.config['num_workers']

	def setup(self, stage: str):

		train_grid = Three_Grid_Dataset(self.train_data_dir, self.property_file, self.grid_size, augmentation = self.augmentation, test = self.test)
		val_grid = Three_Grid_Dataset(self.test_data_dir, self.property_file, self.grid_size, augmentation = self.augmentation, test = self.test)

		train_size = len(train_grid)
		val_size = len(val_grid)


		if self.config["train"]:
  
			self.grid_train = train_grid
			self.grid_val = val_grid

		else:
			self.grid_test = val_grid

	def train_dataloader(self):
		return DataLoader(self.grid_train, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)

	def val_dataloader(self):
		return DataLoader(self.grid_val, batch_size=self.batch_size, num_workers = self.num_workers)

	def test_dataloader(self):
		return DataLoader(self.grid_test, batch_size=self.batch_size, num_workers = self.num_workers)

