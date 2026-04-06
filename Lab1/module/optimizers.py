import numpy as np
from .core import *

class SGD(Optimizer):
	def __init__(self, *modules, lr=0.01):
		super().__init__(*modules)
		self.lr = lr

	def step(self):
		for layer in self._iter_param_layers():
			params = layer.parameters()
			grads = layer.gradients()
			for param_name in params.keys():
				param = params[param_name]
				grad = grads[param_name]
				param -= self.lr * grad

	def state_dict(self):
		return {'lr': self.lr}

	def load_state_dict(self, state_dict):
		self.lr = state_dict['lr']

class SGDM(Optimizer):
	def __init__(self, *modules, lr=0.01, momentum=0.0):
		super().__init__(*modules)
		self.lr = lr
		self.momentum = momentum
		self.v = {}

	def step(self):
		param_to_path = {id(param): path for path, param in self._iter_params_with_paths()}
		for layer in self._iter_param_layers():
			params = layer.parameters()
			grads = layer.gradients()
			for param_name in params.keys():
				param = params[param_name]
				grad = grads[param_name]
				path = param_to_path[id(param)]
				v = self.v.setdefault(path, np.zeros_like(grad))
				v[:] = self.momentum * v + grad
				param -= self.lr * v

	def state_dict(self):
		return {
			'lr': self.lr,
			'momentum': self.momentum,
			'v': {path: v.copy() for path, v in self.v.items()}
		}

	def load_state_dict(self, state_dict):
		self.lr = state_dict['lr']
		self.momentum = state_dict['momentum']
		self.v = {path: v.copy() for path, v in state_dict['v'].items()}

class Adam(Optimizer):
	def __init__(self, *modules, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
		super().__init__(*modules)
		self.lr = lr
		self.beta1 = beta1
		self.beta2 = beta2
		self.eps = eps

		self.m = {}
		self.v = {}
		self.t = 0

	def step(self):
		self.t += 1

		param_to_path = {id(param): path for path, param in self._iter_params_with_paths()}
		for layer in self._iter_param_layers():
			params = layer.parameters()
			grads = layer.gradients()
			for param_name in params.keys():
				param = params[param_name]
				grad = grads[param_name]
				path = param_to_path[id(param)]
				m = self.m.setdefault(path, np.zeros_like(grad))
				v = self.v.setdefault(path, np.zeros_like(grad))
				m[:] = self.beta1 * m + (1.0 - self.beta1) * grad
				v[:] = self.beta2 * v + (1.0 - self.beta2) * (grad ** 2)
				mc = m / (1 - self.beta1 ** self.t)
				vc = v / (1 - self.beta2 ** self.t)
				param -= self.lr * mc / (np.sqrt(vc) + self.eps)

	def state_dict(self):
		return {
			'lr': self.lr,
			'beta1': self.beta1,
			'beta2': self.beta2,
			'eps': self.eps,
			'm': {path: m.copy() for path, m in self.m.items()},
			'v': {path: v.copy() for path, v in self.v.items()},
			't': self.t
		}

	def load_state_dict(self, state_dict):
		self.lr = state_dict['lr']
		self.beta1 = state_dict['beta1']
		self.beta2 = state_dict['beta2']
		self.eps = state_dict['eps']
		self.m = {path: m.copy() for path, m in state_dict['m'].items()}
		self.v = {path: v.copy() for path, v in state_dict['v'].items()}
		self.t = state_dict['t']