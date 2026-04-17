import module as nn

class SineModel(nn.Module):
	def __init__(self, hidden_dims=(16, 16, 16), activation=nn.ReLU, random_policy='He'):
		super().__init__()
		self.layers = []
		dims = [1] + list(hidden_dims) + [1]
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1], random_policy=random_policy))
			if i < len(dims) - 2:
				self.layers.append(activation())

	def forward(self, x):
		x = x.reshape(-1, 1)
		for layer in self.layers:
			x = layer(x)
		x = x.reshape(-1)
		return x

	def backward(self, d):
		d = d.reshape(-1, 1)
		for layer in reversed(self.layers):
			d = layer.backward(d)
		return d
