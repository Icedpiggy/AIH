import module as nn

class ImageModel(nn.Module):
	def __init__(self, dropout_rate=0.2):
		super().__init__()
		self.layers = [
			nn.Linear(784, 256, random_policy='He'),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(dropout_rate),

			nn.Linear(256, 256, random_policy='He'),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(dropout_rate),

			nn.Linear(256, 256, random_policy='He'),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(dropout_rate),

			nn.Linear(256, 12, random_policy='He')
		]

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def backward(self, d):
		for layer in reversed(self.layers):
			d = layer.backward(d)
		return d
