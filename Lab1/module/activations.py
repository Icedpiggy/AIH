from utils.backend import np
from .core import *

class ReLU(Module):
	def __init__(self):
		super().__init__()
		self.mask = None

	def forward(self, x):
		self.mask = x > 0
		return np.where(self.mask, x, 0)

	def backward(self, d):
		d = np.where(self.mask, d, 0)
		self.mask = None
		return d


class Sigmoid(Module):
	def __init__(self):
		super().__init__()
		self.z = None

	def forward(self, x):
		self.z = 1 / (1 + np.exp(-x))
		return self.z

	def backward(self, d):
		return d * (self.z * (1 - self.z))