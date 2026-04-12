import module as nn

class CNNModel(nn.Module):
	def __init__(self, num_classes=12, dropout_rate=0.3):
		super().__init__()

		self.conv1 = [
			nn.Conv2d(1, 32, kernel_size=3, padding=1, random_policy='He'),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1, random_policy='He'),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(dropout_rate)
		]

		self.conv2 = [
			nn.Conv2d(32, 64, kernel_size=3, padding=1, random_policy='He'),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1, random_policy='He'),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d(2, 2),
			nn.Dropout(dropout_rate)
		]

		self.fc = [
			nn.Linear(7 * 7 * 64, 256, random_policy='He'),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(256, 128, random_policy='He'),
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout(dropout_rate),
			nn.Linear(128, num_classes, random_policy='He')
		]

	def forward(self, x):
		batch_size = x.shape[0]
		x = x.reshape(batch_size, 1, 28, 28)

		for layer in self.conv1:
			x = layer(x)

		for layer in self.conv2:
			x = layer(x)

		x = x.reshape(batch_size, -1)

		for layer in self.fc:
			x = layer(x)

		return x

	def backward(self, d):
		d = d.reshape(d.shape[0], -1)

		for layer in reversed(self.fc):
			d = layer.backward(d)

		d = d.reshape(d.shape[0], 64, 7, 7)

		for layer in reversed(self.conv2):
			d = layer.backward(d)

		for layer in reversed(self.conv1):
			d = layer.backward(d)

		return d.reshape(d.shape[0], -1)
