import numpy as np
from .core import *
from .init import *

class Linear(Module):
	def __init__(self, input_dim, output_dim, random_policy=None):
		super().__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.w = None
		self.b = None

		self.w = random_array(input_dim, output_dim, random_policy=random_policy)
		self.b = np.zeros(self.output_dim)

		self.dw = np.zeros_like(self.w)
		self.db = np.zeros_like(self.b)
		self.x = None

	def parameters(self):
		return {'w': self.w, 'b': self.b}

	def gradients(self):
		return {'w': self.dw, 'b': self.db}

	def forward(self, x):
		self.x = x
		return x @ self.w + self.b

	def backward(self, d):
		self.dw += self.x.T @ d
		self.db += d.sum(axis=0)
		d = d @ self.w.T
		self.x = None
		return d

class Dropout(Module):
	def __init__(self, dropout_rate):
		super().__init__()
		self.dropout_rate = dropout_rate
		self.mask = None

	def forward(self, x):
		if self.training:
			self.mask = np.random.rand(*x.shape) > self.dropout_rate
			return np.where(self.mask, x, 0) / (1.0 - self.dropout_rate)
		else:
			return x

	def backward(self, d):
		d = np.where(self.mask, d, 0) / (1.0 - self.dropout_rate)
		self.mask = None
		return d

class BatchNorm(Module):
	def __init__(self, num_features, momentum=0.1, eps=1e-5):
		super().__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.eps = eps

		self.gamma = np.ones(num_features)
		self.beta = np.zeros(num_features)

		self.running_mean = np.zeros(num_features)
		self.running_var = np.zeros(num_features)

		self.d_gamma = np.zeros(num_features)
		self.d_beta = np.zeros(num_features)

		self.x_centered = None
		self.x_normed = None
		self.batch_mean = None
		self.batch_var = None

	def parameters(self):
		return {'gamma': self.gamma, 'beta': self.beta}

	def gradients(self):
		return {'gamma': self.d_gamma, 'beta': self.d_beta}

	def forward(self, x):
		if self.training:
			self.batch_mean = x.mean(axis=0)
			self.batch_var = x.var(axis=0)

			self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
			self.running_var = (1 - self.momentum) * self.running_var + self.momentum * self.batch_var

			mean = self.batch_mean
			var = self.batch_var
		else:
			mean = self.running_mean
			var = self.running_var

		self.x_centered = x - mean
		self.x_normed = self.x_centered / np.sqrt(var + self.eps)
		return self.gamma * self.x_normed + self.beta

	def backward(self, d):
		batch_size = d.shape[0]

		self.d_gamma = np.sum(d * self.x_normed, axis=0)
		self.d_beta = d.sum(axis=0)

		std = np.sqrt(self.batch_var + self.eps)
		d_normed = d * self.gamma
		d_var = np.sum(d_normed * self.x_centered, axis=0) * (-0.5) / (std ** 3)
		d_mean = np.sum(d_normed, axis=0) / (-std) + d_var * np.mean(-2 * self.x_centered, axis=0)
		d = d_normed / std + d_var * 2 * self.x_centered / batch_size + d_mean / batch_size

		self.x_centered = None
		self.x_normed = None
		self.batch_mean = None
		self.batch_var = None
		return d