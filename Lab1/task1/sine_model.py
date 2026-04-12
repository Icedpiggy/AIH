import module as nn

class SineModel(nn.Module):
	def __init__(self, hidden_dim=16):
		super().__init__()
		self.layers = [
			nn.Linear(1, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1, random_policy='He')
		]

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
