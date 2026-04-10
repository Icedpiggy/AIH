import numpy as np
from .core import *
from numpy.lib.stride_tricks import as_strided

class Conv2d(Module):
	pass

class MaxPool2d(Module):
	def __init__(self, kernel_size, stride=None):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride else kernel_size
		self.x_shape = None
		self.max_idx = None

	def forward(self, x):
		self.x_shape = x.shape
		batch_size, channels, H, W = x.shape
		out_h = (H - self.kernel_size) // self.stride + 1
		out_w = (W - self.kernel_size) // self.stride + 1
		windows = as_strided(
			x,
			shape=(batch_size, channels, out_h, out_w, self.kernel_size, self.kernel_size),
			strides=(x.strides[0], x.strides[1], x.strides[2] * self.stride, x.strides[3] * self.stride, x.strides[2], x.strides[3])
		)
		windows = windows.reshape(windows.shape[0], windows.shape[1], windows.shape[2], windows.shape[3], -1)
		max_val = np.max(windows, axis=-1)
		self.max_idx = np.argmax(windows, axis=-1)
		return max_val

	def backward(self, d):
		batch_size, channels, out_h, out_w = d.shape
		batch_size, channels, H, W = self.x_shape
		out_idx_bias = np.arange(H * W).reshape(H, W)[::self.stride, ::self.stride]
		kernel_idx_bias = np.arange(self.kernel_size).reshape(-1, 1) * W + np.arange(self.kernel_size).reshape(1, -1)
		in_idx = out_idx_bias.reshape(out_h, out_w, 1) + kernel_idx_bias.reshape(1, 1, -1)
		
		in_idx = np.broadcast_to(in_idx, (batch_size, channels, out_h, out_w, self.kernel_size * self.kernel_size))
		in_max_idx = in_idx[
			np.arange(batch_size)[:, None, None, None],
			np.arange(channels)[None, :, None, None],
			np.arange(out_h)[None, None, :, None],
			np.arange(out_w)[None, None, None, :],
			self.max_idx
   		]
		in_max_idx = in_max_idx.reshape(batch_size, channels, -1)
		
		d_out = np.zeros((batch_size, channels, H * W))
		d_out[
			np.arange(batch_size)[:, None, None],
			np.arange(channels)[None, :, None],
			in_max_idx
		] += d.reshape(batch_size, channels, -1)[:, :, :]
		d_out = d_out.reshape(batch_size, channels, H, W)

		self.x_shape = None
		self.max_idx = None
		return d_out