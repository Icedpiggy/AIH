from utils.backend import np
from .core import *

class CrossEntropyLoss(Module):
	def __init__(self):
		self.y = None
		self.z = None
		return

	def forward(self, x, y):
		x_shifted = x - np.max(x, axis=-1, keepdims=True)
		exp_x = np.exp(x_shifted)
		self.y = y
		self.z = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
		loss = -np.log(self.z[np.arange(y.shape[0]), y] + 1e-10)
		return loss.mean()

	def backward(self):
		batch_size = self.y.shape[0]
		d = self.z.copy()
		d[np.arange(batch_size), self.y] -= 1
		d /= batch_size
		self.y = None
		self.z = None
		return d

class MSELoss(Module):
	def __init__(self):
		self.z = None

	def forward(self, x, y):
		self.z = x - y
		return (self.z ** 2).mean()

	def backward(self):
		d = 2 * self.z / self.z.size
		self.z = None
		return d

class MAELoss(Module):
	def __init__(self):
		self.z = None

	def forward(self, x, y):
		self.z = x - y
		return np.abs(self.z).mean()

	def backward(self):
		d = np.sign(self.z) / self.z.size
		self.z = None
		return d
