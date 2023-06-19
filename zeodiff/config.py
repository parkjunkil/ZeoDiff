from sacred import Experiment

ex = Experiment("ddpm", save_git_info=False)


@ex.config
def config():

	exp_name = "ddpm"

	seed = 42
	train = True
	precision = 16

	# ddpm model
	# (UNET parameter)
	dim = 32
	dim_mults = (1,2,4)
	channels = 3 
	self_condition = False
	target_prop = None
	# (Diffusion parameter)
	timesteps = 1500
	loss_type = "huber"


	# cell parameter prediction model
	c_model_dir = "models/lattice_regressor.ckpt"
	c_dim_mults = (1,2,4)

	# data
	model_dir = "models/"
	train_dataset = "../data/training/"
	test_dataset = "../data/test/"
	batch_size = 128 # batch size per gpu
	num_workers = 8
	augmentation = True # apply augmentation on database (rotation and translation)
	test_only_100 = False # test trial on dataset of size 100
	property_file = "../data/properties.pickle"


	# training
	accelerator = "gpu"
	n_gpu = 4
	devices = 4
	num_nodes = 1
	optimizer = "adam"
	lr = 1e-4
	log_dir = "../logs/"
	max_epochs = 2000
	n_iter = 10000000
	save_dir = "models/" # where models designated by callbacks will be stored
	grid_size = 32
	strategy = "ddp" # DDPStrategy(find_unused_parameters=True) is now being used as default. If you want to change it, modify trainer part of run.py.
	early_stopping = 50
	load_model = None


	# evaluation
	eval_model = "unconditional.ckpt"
	sample_dir = "/samples/"
	sample_freq = 200
	target_value = 0.05
	n_sample = 10000
