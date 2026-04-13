from utils.backend import np

class Module:
	def __init__(self):
		self.training = True

	def parameters(self):
		return {}

	def buffers(self):
		return {}

	def gradients(self):
		return {}

	@property
	def has_parameters(self):
		params = self.parameters()
		return bool(params and any(v is not None for v in params.values()))

	def children(self):
		for attr_name in dir(self):
			if attr_name.startswith('_'):
				continue
			attr = getattr(self, attr_name)
			if isinstance(attr, Module):
				yield attr
			elif isinstance(attr, list):
				for item in attr:
					if isinstance(item, Module):
						yield item

	def named_children(self):
		for attr_name in dir(self):
			if attr_name.startswith('_'):
				continue
			attr = getattr(self, attr_name)
			if isinstance(attr, Module):
				yield attr_name, attr
			elif isinstance(attr, list):
				for idx, item in enumerate(attr):
					if isinstance(item, Module):
						yield f'{attr_name}.{idx}', item

	def forward(self, *args, **kwargs):
		raise NotImplementedError(f"{self.__class__.__name__}.forward() not implemented")

	def backward(self, *args, **kwargs):
		raise NotImplementedError(f"{self.__class__.__name__}.backward() not implemented")

	def train(self):
		self.training = True
		for child in self.children():
			child.train()

	def eval(self):
		self.training = False
		for child in self.children():
			child.eval()

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)

	def state_dict(self, prefix=''):
		state = {}
		params = self.parameters()
		for key, value in params.items():
			if value is not None:
				state_key = prefix + key
				state[state_key] = value.copy() if isinstance(value, np.ndarray) else value

		buffers = self.buffers()
		for key, value in buffers.items():
			if value is not None:
				state_key = prefix + key
				state[state_key] = value.copy() if isinstance(value, np.ndarray) else value

		for name, child in self.named_children():
			child_prefix = prefix + name + '.'
			state.update(child.state_dict(child_prefix))

		return state

	def load_state_dict(self, state_dict, prefix=''):
		params = self.parameters()
		for key, value in params.items():
			state_key = prefix + key
			if state_key in state_dict and value is not None:
				if isinstance(value, np.ndarray):
					value[:] = np.array(state_dict[state_key])
				else:
					setattr(self, key, state_dict[state_key])

		buffers = self.buffers()
		for key, value in buffers.items():
			state_key = prefix + key
			if state_key in state_dict and value is not None:
				if isinstance(value, np.ndarray):
					value[:] = np.array(state_dict[state_key])
				else:
					setattr(self, key, state_dict[state_key])

		for name, child in self.named_children():
			child_prefix = prefix + name + '.'
			child.load_state_dict(state_dict, child_prefix)

class Optimizer:
	def __init__(self, *modules):
		self.modules = modules

	def _iter_params_with_paths(optimizer):
		for idx, layer in enumerate(optimizer._iter_param_layers()):
			params = layer.parameters()
			for param_name, param in params.items():
				if param is not None:
					yield f'layer{idx}.{param_name}', param

	def _iter_param_layers(self, module=None):
		if module is None:
			for root in self.modules:
				yield from self._iter_param_layers(root)
		else:
			if module.has_parameters:
				yield module
			for child in module.children():
				yield from self._iter_param_layers(child)

	def step(self):
		raise NotImplementedError(f"{self.__class__.__name__}.step() not implemented")

	def zero_grad(self):
		for layer in self._iter_param_layers():
			for grad in layer.gradients().values():
				if grad is not None:
					grad[:] = 0

	def state_dict(self):
		return {}

	def load_state_dict(self, state_dict):
		pass
