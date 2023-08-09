import torch
import numpy as np
from torch import  nn
from torch.nn import functional as F
from tqdm import tqdm
from util import write_visit_sample

def linear_beta_schedule(timesteps):
	beta_start = 0.0001
	beta_end = 0.02
	return torch.linspace(beta_start, beta_end, timesteps)



def extract(a, t, x_shape):
	batch_size = t.shape[0]
	out = a.gather(-1, t.cpu())
	return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion(nn.Module):
	def __init__(self, betas, loss_type='huber'):
		super().__init__()

		# define beta schedule and alphas (parameters regarding variance of noising process)
		betas              = betas.type(torch.float64)
		timesteps          = betas.shape[0]
		self.timesteps = int(timesteps)
		self.loss_type = loss_type

		alphas = 1. - betas
		alphas_cumprod = torch.cumprod(alphas, axis=0)
		alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
		sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # noising process
		sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
		sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

		posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

		self.betas = betas
		self.sqrt_alphas_cumprod = sqrt_alphas_cumprod
		self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod
		self.posterior_variance = posterior_variance
		self.sqrt_recip_alphas = sqrt_recip_alphas


        
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)

		sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
		sqrt_one_minus_alphas_cumprod_t = extract(
			self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
		)

		return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

	@torch.no_grad()
	def p_sample(self, model, x, t, t_index, context=None):
   
		if context is not None:
			context = context.to(t.device)
		betas_t = extract(self.betas, t, x.shape)
		sqrt_one_minus_alphas_cumprod_t = extract(
			self.sqrt_one_minus_alphas_cumprod, t, x.shape
		)
		sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
		# Predict the mean of denoisng process
		model_mean = sqrt_recip_alphas_t * (
			x - betas_t * model(x, t, context) / sqrt_one_minus_alphas_cumprod_t
		)

		if t_index == 0:
			return model_mean
		else:
			posterior_variance_t = extract(self.posterior_variance, t, x.shape)
			noise = torch.randn_like(x)
          
			return model_mean + torch.sqrt(posterior_variance_t) * noise 

	@torch.no_grad()
	def p_sample_loop(self, model, shape, context=None):
		device = next(model.parameters()).device
            
		b = shape[0]
		# sampling starts from pure random noise
		img = torch.randn(shape, device=device)
	    
		imgs = []

		for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
			img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, context)
			imgs.append(img.cpu().numpy())
		return imgs

	@torch.no_grad()
	def sample(self, model, grid_size, batch_size, channels, context=None):
		return np.array(self.p_sample_loop(model, shape=(batch_size, channels, grid_size, grid_size, grid_size),context=context))
   
	def progressive_sample(self, model, cell_model, directory,  shape, channels=3, context=None):
		device = next(model.parameters()).device

		b = shape[0]

		img = torch.randn(shape, device=device)

		imgs = []

		for i in tqdm(reversed(range(0, self.timesteps)), desc='smpling loop time step', total=self.timesteps):
			img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long),i,context)
			imgs.append(img.cpu().numpy())
		
		cell_param = cell_model(torch.tensor(imgs[-1]))

		cell_param = [j*100 for j in cell_param]

		for i in range(len(imgs)):
			data = imgs[i].reshape(channels, grid_size, grid_size, grid_size)
			write_visit_sample(data, cell=cell_param, stem='sample_'+str(i), save_dir = directory)	 


	@torch.no_grad()
	def large_sample(self, model, cell_model, num_sample, directory, target_value=None, grid_size = 32, channels = 3):
		num_50_cycle = num_sample // 50
		left_over = num_sample - num_50_cycle*50

		count = 1
		for _ in range(num_50_cycle):
				if target_value is not None:
					context = torch.tensor([target_value]).repeat(50, channels, grid_size, grid_size, grid_size)
				samples = self.sample(model, grid_size=grid_size, batch_size = 50, channels=channels, context = context)[-1]
				cell_param_list = cell_model(torch.tensor(samples))

				for i in range(len(samples)):
						arr = samples[i].reshape(channels,grid_size, grid_size, grid_size)
						cell_param = [j*100 for j in cell_param_list[i]]
						write_visit_sample(arr, cell = cell_param, stem = 'sample_'+str(count), save_dir=directory)
						count += 1

		if target_value is not None:
			context = torch.tensor([target_value]).repeat(left_over, channels, grid_size, grid_size, grid_size)
		samples = self.sample(model, grid_size=grid_size, batch_size = left_over, channels=channels, context = context)[-1]
		cell_param_list = cell_model(torch.tensor(samples))

		for i in range(len(samples)):
				arr = samples[i].reshape(channels,grid_size, grid_size, grid_size)
				cell_param = [j*100 for j in cell_param_list[i]]
				write_visit_sample(arr, cell = cell_param, stem = 'sample_'+str(count), save_dir=directory)
				count += 1

	def training_losses(self, model, x_start, t, c=None, noise=None):

		if noise is None:
			noise = torch.randn_like(x_start)

		x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

		predicted_noise = model(x_t, t, c)

		# Calculate the loss (huber loss encouraged to use)
		if self.loss_type == 'huber':
			loss = F.smooth_l1_loss(noise, predicted_noise)             

		elif self.loss_type == 'mae':
			loss = F.l1_loss(noise, predicted_noise)

		elif self.loss_type == 'mse':
			loss = F.mse_loss(noise, predicted_noise)

		else:
			raise NotImplementedError(self.loss_type)

		return loss
