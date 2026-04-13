import module as nn

class CNNModel(nn.Module):
	def __init__(self, conv_channels=((32, 32), (64, 64)), activation=nn.ReLU,
				 dropout_rate=0.3, init='He', bn_momentum=0.1,
				 num_classes=12, fc_dims=(256, 128)):
		super().__init__()
		self.conv_layers = []
		self.fc_layers = []

		in_ch = 1
		for block_chs in conv_channels:
			for out_ch in block_chs:
				self.conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, random_policy=init))
				self.conv_layers.append(nn.BatchNorm2d(out_ch, momentum=bn_momentum))
				self.conv_layers.append(activation())
				in_ch = out_ch
			self.conv_layers.append(nn.MaxPool2d(2, 2))
			self.conv_layers.append(nn.Dropout(dropout_rate))

		num_pools = len(conv_channels)
		spatial = 28 // (2 ** num_pools)
		fc_in = spatial * spatial * in_ch

		dims = [fc_in] + list(fc_dims) + [num_classes]
		for i in range(len(dims) - 1):
			self.fc_layers.append(nn.Linear(dims[i], dims[i + 1], random_policy=init))
			if i < len(dims) - 2:
				self.fc_layers.append(nn.BatchNorm1d(dims[i + 1], momentum=bn_momentum))
				self.fc_layers.append(activation())
				self.fc_layers.append(nn.Dropout(dropout_rate))

	def forward(self, x):
		batch_size = x.shape[0]
		x = x.reshape(batch_size, 1, 28, 28)
		for layer in self.conv_layers:
			x = layer(x)
		self._spatial_shape = x.shape[1:]
		x = x.reshape(batch_size, -1)
		for layer in self.fc_layers:
			x = layer(x)
		return x

	def backward(self, d):
		for layer in reversed(self.fc_layers):
			d = layer.backward(d)
		d = d.reshape(d.shape[0], *self._spatial_shape)
		for layer in reversed(self.conv_layers):
			d = layer.backward(d)
		return d.reshape(d.shape[0], -1)
