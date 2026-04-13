import module as nn

class ImageModel(nn.Module):
	def __init__(self, hidden_dims=(256, 256, 256), activation=nn.ReLU, dropout_rate=0.2, init='He', bn_momentum=0.1):
		super().__init__()
		self.layers = []
		dims = [784] + list(hidden_dims) + [12]
		for i in range(len(dims) - 1):
			self.layers.append(nn.Linear(dims[i], dims[i + 1], random_policy=init))
			if i < len(dims) - 2:
				self.layers.append(nn.BatchNorm1d(dims[i + 1], momentum=bn_momentum))
				self.layers.append(activation())
				self.layers.append(nn.Dropout(dropout_rate))

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def backward(self, d):
		for layer in reversed(self.layers):
			d = layer.backward(d)
		return d
