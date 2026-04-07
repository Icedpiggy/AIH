import os
import numpy as np
import pickle
from PIL import Image
from dataprocess import *

class ImgDataset(Dataset):
	def __init__(self, x=None, y=None):
		super().__init__()
		self.x = x
		self.y = y

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

	def state_dict(self):
		return {
			'x': self.x, 'y': self.y
		}

	def load_state_dict(self, data):
		self.x = data['x']
		self.y = data['y']

def load_data(data_dir):
	x = []
	y = []
	categories = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

	for category in categories:
		category_path = os.path.join(data_dir, category)
		files = [f for f in os.listdir(category_path) if f.endswith('.bmp')]

		for filename in files:
			img_path = os.path.join(category_path, filename)
			img = Image.open(img_path)
			img_array = np.array(img)
			if len(img_array.shape) == 3:
				img_array = img_array[:,:,0]
			x.append(img_array)
			y.append(int(category) - 1)

	return np.array(x), np.array(y)

if __name__ == "__main__":
	np.random.seed(42)
	train_rate = 0.9

	data_dir = './data_2/raw'
	x, y = load_data(data_dir)

	train_indices = []
	val_indices = []

	for i in range(12):
		class_indices = np.where(y == i)[0]
		np.random.shuffle(class_indices)
		split_idx = int(len(class_indices) * train_rate)

		train_indices.extend(class_indices[:split_idx])
		val_indices.extend(class_indices[split_idx:])

	train_x = x[train_indices]
	train_y = y[train_indices]
	val_x = x[val_indices]
	val_y = y[val_indices]

	os.makedirs('./data_2', exist_ok=True)

	with open('./data_2/train.pkl', 'wb') as f:
		pickle.dump({'x': train_x, 'y': train_y}, f)

	with open('./data_2/val.pkl', 'wb') as f:
		pickle.dump({'x': val_x, 'y': val_y}, f)

	print(f'Train set: {len(train_x)} samples, shape: {train_x.shape}')
	print(f'Val set: {len(val_x)} samples, shape: {val_x.shape}')
	print()
	print('Class distribution:')
	for i in range(12):
		train_count = np.sum(train_y == i)
		val_count = np.sum(val_y == i)
		total_count = train_count + val_count
		train_ratio = train_count / total_count * 100 if total_count > 0 else 0
		val_ratio = val_count / total_count * 100 if total_count > 0 else 0
		print(f'Class {i}: train={train_count} ({train_ratio:.2f}%), val={val_count} ({val_ratio:.2f}%)')