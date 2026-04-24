import os
from utils.backend import numpy
from PIL import Image
from utils import Dataset, save_dataset

class ImgDataset(Dataset):
	def __init__(self, x, y, training=False):
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

		if numpy.random.rand() < 0.5:
			img = self.random_rotation(img)

		if numpy.random.rand() < 0.5:
			img = self.random_scale(img)

		if numpy.random.rand() < 0.5:
			img = self.random_shift(img)

		if numpy.random.rand() < 0.3:
			img = self.random_cutout(img)

		if numpy.random.rand() < 0.4:
			img = self.gaussian_filter(img)

		if numpy.random.rand() < 0.3:
			img = self.add_gaussian_noise(img)

		return img

	def random_rotation(self, img):
		from scipy.ndimage import rotate
		angle = numpy.random.uniform(-10, 10)
		return rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0, prefilter=False)

	def random_shift(self, img):
		from scipy.ndimage import shift
		shift_y = numpy.random.uniform(-2, 2)
		shift_x = numpy.random.uniform(-2, 2)
		return shift(img, [shift_y, shift_x], order=1, mode='constant', cval=0.0, prefilter=False)

	def random_scale(self, img):
		from scipy.ndimage import zoom
		scale = numpy.random.uniform(0.9, 1.1)
		h, w = img.shape
		zoomed = zoom(img, scale, order=1, prefilter=False)

		if scale > 1:
			start_h = (zoomed.shape[0] - h) // 2
			start_w = (zoomed.shape[1] - w) // 2
			return zoomed[start_h:start_h+h, start_w:start_w+w]
		else:
			pad_h = (h - zoomed.shape[0]) // 2
			pad_w = (w - zoomed.shape[1]) // 2
			padded = numpy.zeros((h, w))
			padded[pad_h:pad_h+zoomed.shape[0], pad_w:pad_w+zoomed.shape[1]] = zoomed
			return padded

	def add_gaussian_noise(self, img):
		noise = numpy.random.normal(0, 0.02, img.shape)
		return numpy.clip(img + noise, 0, 1)

	def random_cutout(self, img):
		h, w = img.shape
		size = max(1, int(h * 0.1))
		y = numpy.random.randint(0, h - size)
		x = numpy.random.randint(0, w - size)
		img = img.copy()
		img[y:y+size, x:x+size] = 0
		return img

	def gaussian_filter(self, img):
		from scipy.ndimage import gaussian_filter
		sigma = numpy.random.uniform(0.3, 1.0)
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
			img_array = numpy.array(img)
			if len(img_array.shape) == 3:
				img_array = img_array[:,:,0]
			x.append(1.0 - img_array)
			y.append(int(category) - 1)

	return numpy.array(x), numpy.array(y)


def prepare_image_data(src_dir, dst_dir, train_rate=0.8, num_classes=12):
	train_pkl = os.path.join(dst_dir, 'train.pkl')
	val_pkl = os.path.join(dst_dir, 'val.pkl')

	if os.path.exists(train_pkl) and os.path.exists(val_pkl):
		return

	x, y = load_data(src_dir)

	train_indices = []
	val_indices = []

	for i in range(num_classes):
		class_indices = numpy.where(y == i)[0]
		numpy.random.shuffle(class_indices)
		split_idx = int(len(class_indices) * train_rate)

		train_indices.extend(class_indices[:split_idx])
		val_indices.extend(class_indices[split_idx:])

	save_dataset(ImgDataset(x[train_indices], y[train_indices], training=True), dst_dir, 'train')
	save_dataset(ImgDataset(x[val_indices], y[val_indices], training=False), dst_dir, 'val')


if __name__ == "__main__":
	HERE = os.path.dirname(os.path.abspath(__file__))
	numpy.random.seed(42)

	data_dir = os.path.join(HERE, '..', 'data_2')
	x, y = load_data(data_dir)

	train_indices = []
	val_indices = []

	for i in range(12):
		class_indices = numpy.where(y == i)[0]
		numpy.random.shuffle(class_indices)
		split_idx = int(len(class_indices) * 0.9)

		train_indices.extend(class_indices[:split_idx])
		val_indices.extend(class_indices[split_idx:])

	train_x = x[train_indices]
	train_y = y[train_indices]
	val_x = x[val_indices]
	val_y = y[val_indices]

	data_dir = os.path.join(HERE, '..', 'data_2_test')
	test_x, test_y = load_data(data_dir)

	data_dir = os.path.join(HERE, 'data')
	os.makedirs(data_dir, exist_ok=True)
	save_dataset(ImgDataset(train_x, train_y, training=True), data_dir, 'train')
	save_dataset(ImgDataset(val_x, val_y, training=False), data_dir, 'val')
	save_dataset(ImgDataset(test_x, test_y, training=False), data_dir, 'test')

	print(f'Train set: {len(train_x)} samples')
	print(f'Val set: {len(val_x)} samples')
	print(f'Test set: {len(test_x)} samples')
	print()
	print('Class distribution:')
	for i in range(12):
		train_count = numpy.sum(train_y == i)
		val_count = numpy.sum(val_y == i)
		test_count = numpy.sum(test_y == i)
		print(f'Class {i}: train={train_count}, val={val_count}, test={test_count}')
