import numpy as np
from .core import *
from numpy.lib.stride_tricks import as_strided

class Conv2d(Module):
	def __init__(self):
		pass
	def im2col(self, x):
		pass
	def col2im(self, x):
		pass
	def forward(self, x):
		pass
	def backward(self, x):
		pass


class MaxPool2d(Module):
	def __init__(self, kernel_size, stride=None):
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride else kernel_size
		self.H = None
		self.W = None
		self.max_idx = None

	def forward(self, x):
		batch_size, channels, H, W = x.shape
		out_h = (H - self.kernel_size) // self.stride + 1
		out_w = (W - self.kernel_size) // self.stride + 1
		windows = as_strided(
			x,
			shape=(batch_size, channels, out_h, out_w, self.kernel_size, self.kernel_size),
			strides=(x.strides[0], x.strides[1], x.strides[2] * self.stride, x.strides[3] * self.stride, x.strides[2], x.strides[3])
		)
		windows = windows.reshape(batch_size, channels, out_h, out_w, -1)
		max_val = np.max(windows, axis=-1)
		if self.training:
			self.H, self.W = H, W
			self.max_idx = np.argmax(windows, axis=-1)
		return max_val

	def backward(self, d):
		batch_size, channels, out_h, out_w = d.shape
		H, W = self.H, self.W

		h_kernel = self.max_idx // self.kernel_size
		w_kernel = self.max_idx % self.kernel_size

		n_idx = np.broadcast_to(np.arange(batch_size).reshape(-1, 1, 1, 1), d.shape)
		c_idx = np.broadcast_to(np.arange(channels).reshape(1, -1, 1, 1), d.shape)
		h_idx = np.arange(out_h).reshape(1, 1, -1, 1) * self.stride + h_kernel
		w_idx = np.arange(out_w).reshape(1, 1, 1, -1) * self.stride + w_kernel
		
		n_flat = n_idx.flatten()
		c_flat = c_idx.flatten()
		h_flat = h_idx.flatten()
		w_flat = w_idx.flatten()
		d_flat = d.flatten()

		d_out = np.zeros((batch_size, channels, H, W))
		d_out[n_flat, c_flat, h_flat, w_flat] = d_flat

		self.x_shape = None
		self.max_idx = None
		return d_out