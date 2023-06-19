import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np

def exists(x):
	return x is not None

def default(val, d):
	if exists(val):
		return val
	return d() if isfunction(d) else d

class Residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn

	def forward(self, x, *args, **kwargs):
		return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
	return nn.Sequential(
		nn.Upsample(scale_factor=2, mode="nearest"),
		nn.Conv3d(dim, default(dim_out, dim), 3, padding=1, padding_mode='circular'),
	)

def Downsample(dim, dim_out=None):
	return nn.Sequential(
		nn.Conv3d(dim, default(dim_out, dim), 3, padding=1,stride=2, padding_mode='circular'),
	)

class SinusoidalPositionEmbeddings(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, time):
		device = time.device
		half_dim = self.dim // 2
		embeddings = math.log(10000) / (half_dim - 1)
		embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
		embeddings = time[:, None] * embeddings[None, :]
		embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
		return embeddings

class WeightStandardizedConv3d(nn.Conv3d):
	def forward(self, x):
		eps = 1e-5 if x.dtype == torch.float32 else 1e-3

		weight = self.weight

		mean = reduce(weight, "o ... -> o 1 1 1 1", "mean")

		var = reduce(weight, "o ... -> o 1 1 1 1", partial(torch.var, unbiased=False))
		normalized_weight = (weight - mean) * (var + eps).rsqrt()

		return F.conv3d(
			x,
			normalized_weight,
			self.bias,
			self.stride,
			self.padding,
			self.dilation,
			self.groups,
		)


class Block(nn.Module):
	def __init__(self, dim, dim_out, groups=8):
		super().__init__()
		self.proj = WeightStandardizedConv3d(dim, dim_out, 3, padding=1, padding_mode='circular')
		self.norm = nn.GroupNorm(groups, dim_out)
		self.act = nn.SiLU()

	def forward(self, x, scale_shift=None):
		x = self.proj(x)
		x = self.norm(x)

		if exists(scale_shift):
			scale, shift = scale_shift
			x = x * (scale + 1) + shift

		x = self.act(x)
		return x


class ResnetBlock(nn.Module):

	def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
		super().__init__()
		self.mlp = (
			nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
			if exists(time_emb_dim)
			else None
		)

		self.block1 = Block(dim, dim_out, groups=groups)
		self.block2 = Block(dim_out, dim_out, groups=groups)
		self.res_conv = nn.Conv3d(dim, dim_out, 3, padding=1, padding_mode='circular') if dim != dim_out else nn.Identity()

	def forward(self, x, time_emb=None):
		scale_shift = None
		if exists(self.mlp) and exists(time_emb):
			time_emb = self.mlp(time_emb)
			time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
			scale_shift = time_emb.chunk(2, dim=1)

		h = self.block1(x, scale_shift=scale_shift)
		h = self.block2(h)
		return h + self.res_conv(x)


class Attention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)
		self.to_out = nn.Conv3d(hidden_dim, dim, 1)

	def forward(self, x):
		b, c, lx, ly, lz  = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
		)
		q = q * self.scale

		sim = einsum("b h d i, b h d j -> b h i j", q, k)
		sim = sim - sim.amax(dim=-1, keepdim=True).detach()
		attn = sim.softmax(dim=-1)

		out = einsum("b h i j, b h d j -> b h i d", attn, v)
		out = rearrange(out, "b h (x y z) d -> b (h d) x y z", x=lx, y=ly, z=lz)
		return self.to_out(out)

class LinearAttention(nn.Module):
	def __init__(self, dim, heads=4, dim_head=32):
		super().__init__()
		self.scale = dim_head**-0.5
		self.heads = heads
		hidden_dim = dim_head * heads
		self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

		self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, 1),nn.GroupNorm(1, dim))

	def forward(self, x):
		b, c, lx, ly, lz = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=1)
		q, k, v = map(
			lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
		)

		q = q.softmax(dim=-2)
		k = k.softmax(dim=-1)

		q = q * self.scale
		context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

		out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
		out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=lx, y=ly, z=lz)
		return self.to_out(out)

class PreNorm(nn.Module):
	def __init__(self, dim, fn):
		super().__init__()
		self.fn = fn
		self.norm = nn.GroupNorm(1, dim)

	def forward(self, x):
		x = self.norm(x)
		return self.fn(x)

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
	mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
	for i in range(fc_num_layers-1):
		mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
	mods += [nn.Linear(hidden_dim, out_dim)]
	return nn.Sequential(*mods)

# U-Net regression model for lattice parameters prediction
class Unet_Reg(nn.Module):
	def __init__(
		self,
		dim,
		init_dim=None,
		out_dim=None,
		dim_mults=(1, 2, 4),
		channels=3,
		self_condition=False,
		resnet_block_groups=4,
	):
		super().__init__()

		self.channels = channels
		self.self_condition = self_condition
		input_channels = channels * (2 if self_condition else 1)

		init_dim = default(init_dim, dim)

		self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding=1, padding_mode='circular') # changed to 1 and 0 from 7,3

		dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
		in_out = list(zip(dims[:-1], dims[1:]))

		block_klass = partial(ResnetBlock, groups=resnet_block_groups)

		time_dim = dim * 4

		self.time_mlp = nn.Sequential(
			SinusoidalPositionEmbeddings(dim),
			nn.Linear(dim, time_dim),
			nn.GELU(),
			nn.Linear(time_dim, time_dim),
		)
		self.downs = nn.ModuleList([])
		self.ups = nn.ModuleList([])
		num_resolutions = len(in_out)

		for ind, (dim_in, dim_out) in enumerate(in_out):
			is_last = ind >= (num_resolutions - 1)

			self.downs.append(
				nn.ModuleList(
					[
						block_klass(dim_in, dim_in),
						Downsample(dim_in, dim_out)
						if not is_last
						else nn.Conv3d(dim_in, dim_out, 3, padding=1, padding_mode='circular'),
					]
				)
			)

		mid_dim = dims[-1]
		self.mid_block1 = block_klass(mid_dim, mid_dim)
		self.mid_block2 = block_klass(mid_dim, mid_dim)

		for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
			is_last = ind == (len(in_out) - 1)

			self.ups.append(
				nn.ModuleList(
					[
						block_klass(dim_out + dim_in, dim_out),
						Upsample(dim_out, dim_in)
						if not is_last
						else nn.Conv3d(dim_out, dim_in, 3, padding=1,  padding_mode='circular'),
					]
				)
			)

		self.out_dim = default(out_dim, channels)

		self.final_res_block = block_klass(dim * 2, dim)
		self.glob_pool = nn.AdaptiveAvgPool3d((1,1,1))
		# final layer that outputs lattice parameters
		self.final_mlp = build_mlp(32,32,2,3)

	def forward(self, x, x_self_cond=None):
    
		b,c,l1,l2,l2 = x.shape

		x = x.float()

		if self.self_condition:
			x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
			x = torch.cat((x_self_cond, x), dim=1)

		x = self.init_conv(x)
		r = x.clone()


		h = []

		for block1, downsample in self.downs:
			x = block1(x)

			h.append(x)

			x = downsample(x)

		x = self.mid_block1(x)
		x = self.mid_block2(x)

		for block1, upsample in self.ups:
			x = torch.cat((x, h.pop()), dim=1)
			x = block1(x)

			x = upsample(x)

		x = torch.cat((x, r), dim=1)

		x = self.final_res_block(x)
		x = self.glob_pool(x)
		x = x.squeeze()
		return self.final_mlp(x)
