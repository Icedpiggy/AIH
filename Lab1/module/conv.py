from utils.backend import np, as_strided
from .core import *
from .init import *

class Conv2d(Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, random_policy=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		self.fan_in = in_channels * kernel_size * kernel_size
		self.fan_out = out_channels

		self.w = random_array(self.fan_in, self.fan_out, random_policy=random_policy)
		self.b = np.zeros(out_channels)

		self.dw = np.zeros_like(self.w)
		self.db = np.zeros_like(self.b)

		self.x = None
		self.x_shape = None
		self.col_shape = None
	
	def parameters(self):
		return {'w': self.w, 'b': self.b}
	
	def gradients(self):
		return {'w': self.dw, 'b': self.db}
	
	def im2col(self, x):
		batch_size, channels, H, W = x.shape
		out_h = (H - self.kernel_size) // self.stride + 1
		out_w = (W - self.kernel_size) // self.stride + 1
		windows = as_strided(
			x,
			shape=(batch_size, channels, out_h, out_w, self.kernel_size, self.kernel_size),
			strides=(x.strides[0], x.strides[1], x.strides[2] * self.stride, x.strides[3] * self.stride, x.strides[2], x.strides[3])
		)
		windows = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch_size, out_h, out_w, -1)
		return np.ascontiguousarray(windows)

	def col2im(self, d):
		batch_size, out_h, out_w, _ = d.shape
		d = d.reshape(batch_size, out_h, out_w, self.in_channels, self.kernel_size, self.kernel_size).transpose(0, 3, 1, 2, 4, 5)

		n_idx = np.broadcast_to(np.arange(batch_size).reshape(-1, 1, 1, 1, 1, 1), d.shape)
		c_idx = np.broadcast_to(np.arange(self.in_channels).reshape(1, -1, 1, 1, 1, 1), d.shape)
		h_idx = np.broadcast_to(np.arange(out_h).reshape(1, 1, -1, 1, 1, 1) * self.stride + np.arange(self.kernel_size).reshape(1, 1, 1, 1, -1, 1), d.shape)
		w_idx = np.broadcast_to(np.arange(out_w).reshape(1, 1, 1, -1, 1, 1) * self.stride + np.arange(self.kernel_size).reshape(1, 1, 1, 1, 1, -1), d.shape)

		n_flat = n_idx.flatten()
		c_flat = c_idx.flatten()
		h_flat = h_idx.flatten()
		w_flat = w_idx.flatten()

		d_flat = d.flatten()

		d = np.zeros(self.x_shape)
		d[n_flat, c_flat, h_flat, w_flat] += d_flat
		return d

	def forward(self, x):
		if self.padding > 0:
			x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)
		if self.training:
			self.x_shape = x.shape
		self.x = self.im2col(x)
		x = (self.x @ self.w).transpose(0, 3, 1, 2) + self.b.reshape(1, -1, 1, 1)
		if self.training:
			self.col_shape = self.x.shape
		else:
			self.x = None
		return np.ascontiguousarray(x)
	
	def backward(self, d):
		self.db += d.sum(axis=(0, 2, 3))
		d = np.ascontiguousarray(d.transpose(0, 2, 3, 1).reshape(-1, self.fan_out))
		self.dw += self.x.reshape(-1, self.fan_in).T @ d
		d = d @ self.w.T
		d = d.reshape(self.col_shape)
		d = self.col2im(d)
		if self.padding > 0:
			d = d[:, :, self.padding:-self.padding, self.padding:-self.padding]
		self.x = None
		self.x_shape = None
		return d


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

		self.H = None
		self.W = None
		self.max_idx = None
		return d_out