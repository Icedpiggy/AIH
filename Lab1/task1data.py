import numpy as np
import pickle
from dataprocess import *

class SineDataset(Dataset):
	def __init__(self, x=None, y=None, training=False, eps_x=1e-3, eps_y=1e-3):
		super().__init__()
		self.x = x
		self.y = y
		self.training = training
		self.eps_x = eps_x
		self.eps_y = eps_y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		if self.training:
			x_noisy = self.x[idx] + np.random.normal(0, self.eps_x)
			y_noisy = self.y[idx] + np.random.normal(0, self.eps_y)
			return x_noisy, y_noisy
		else:
			return self.x[idx], self.y[idx]

	def state_dict(self):
		return {
			'x': self.x, 'y': self.y,
			'training': self.training,
			'eps_x': self.eps_x, 'eps_y': self.eps_y
		}

	def load_state_dict(self, data):
		self.x = data['x']
		self.y = data['y']
		self.training = data['training']
		self.eps_x = data['eps_x']
		self.eps_y = data['eps_y']

if __name__ == "__main__":
	np.random.seed(42)
	train_size = 4096
	val_size = 1024

	x = np.random.uniform(-np.pi, np.pi, train_size)
	y = np.sin(x)

	trainset = SineDataset(x, y, training=False, eps_x=1e-3, eps_y=1e-3)
	with open('./data_1/train.pkl', 'wb') as f:
		pickle.dump(trainset.state_dict(), f)

	x = np.random.uniform(-np.pi, np.pi, val_size)
	y = np.sin(x)

	valset = SineDataset(x, y, training=False)
	with open('./data_1/val.pkl', 'wb') as f:
		pickle.dump(valset.state_dict(), f)