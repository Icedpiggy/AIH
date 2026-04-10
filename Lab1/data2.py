import os
import numpy as np
import pickle
from PIL import Image
from dataprocess import *

class ImgDataset(Dataset):
	def __init__(self, x=None, y=None, training=False):
		self.x = x
		self.y = y
		self.training = training

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		img = self.x[idx].copy()
		label = self.y[idx]

		if self.training:
			img = self.apply_augmentation(img)
		img = img.reshape(-1)
		return img, label

	def apply_augmentation(self, img):
		img = img.reshape(28, 28)

		if np.random.rand() < 0.5:
			img = self.random_rotation(img)

		if np.random.rand() < 0.5:
			img = self.random_scale(img)

		if np.random.rand() < 0.5:
			img = self.random_shift(img)

		if np.random.rand() < 0.35:
			img = self.gaussian_filter(img)

		if np.random.rand() < 0.35:
			img = self.adjust_brightness(img)

		if np.random.rand() < 0.35:
			img = self.adjust_contrast(img)

		if np.random.rand() < 0.3:
			img = self.random_cutout(img)

		if np.random.rand() < 0.3:
			img = self.add_gaussian_noise(img)

		return img

	def random_rotation(self, img):
		from scipy.ndimage import rotate
		angle = np.random.uniform(-10, 10)
		return rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0, prefilter=False)

	def random_shift(self, img):
		from scipy.ndimage import shift
		shift_y = np.random.uniform(-2, 2)
		shift_x = np.random.uniform(-2, 2)
		return shift(img, [shift_y, shift_x], order=1, mode='constant', cval=0.0, prefilter=False)

	def random_scale(self, img):
		from scipy.ndimage import zoom
		scale = np.random.uniform(0.9, 1.1)
		h, w = img.shape
		zoomed = zoom(img, scale, order=1, prefilter=False)

		if scale > 1:
			start_h = (zoomed.shape[0] - h) // 2
			start_w = (zoomed.shape[1] - w) // 2
			return zoomed[start_h:start_h+h, start_w:start_w+w]
		else:
			pad_h = (h - zoomed.shape[0]) // 2
			pad_w = (w - zoomed.shape[1]) // 2
			padded = np.zeros((h, w))
			padded[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
			return padded

	def adjust_brightness(self, img):
		delta = np.random.uniform(-0.1, 0.1)
		return np.clip(img + delta, 0, 1)

	def adjust_contrast(self, img):
		factor = np.random.uniform(0.8, 1.2)
		mean = img.mean()
		return np.clip((img - mean) * factor + mean, 0, 1)

	def add_gaussian_noise(self, img):
		noise = np.random.normal(0, 0.02, img.shape)
		return np.clip(img + noise, 0, 1)

	def random_cutout(self, img):
		h, w = img.shape
		size = max(1, int(h * 0.1))
		y = np.random.randint(0, h - size)
		x = np.random.randint(0, w - size)
		img = img.copy()
		img[y:y+size, x:x+size] = 0
		return img

	def gaussian_filter(self, img):
		from scipy.ndimage import gaussian_filter
		sigma = np.random.uniform(0.3, 1.0)
		return gaussian_filter(img, sigma, mode='constant', cval=0.0)

	def state_dict(self):
		return {'x': self.x, 'y': self.y, 'training': self.training}

	def load_state_dict(self, data):
		self.x = data['x']
		self.y = data['y']
		self.training = data['training']

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
			x.append(1.0 - img_array)
			y.append(int(category) - 1)

	return np.array(x), np.array(y)

if __name__ == "__main__":
	np.random.seed(42)
	train_rate = 0.8

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
		pickle.dump({'x': train_x, 'y': train_y, 'training': True}, f)

	with open('./data_2/val.pkl', 'wb') as f:
		pickle.dump({'x': val_x, 'y': val_y, 'training': False}, f)

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