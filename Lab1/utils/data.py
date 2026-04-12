import os
import pickle
import numpy as np


class Dataset:
	def __init__(self):
		pass

	def __len__(self):
		raise NotImplementedError(f"{self.__class__.__name__}.__len__() not implemented")

	def __getitem__(self, idx):
		raise NotImplementedError(f"{self.__class__.__name__}.__getitem__() not implemented")

	def state_dict(self):
		raise NotImplementedError(f"{self.__class__.__name__}.state_dict() not implemented")

	def load_state_dict(self):
		raise NotImplementedError(f"{self.__class__.__name__}.load_state_dict() not implemented")


class Dataloader:
	def __init__(self, dataset, batch_size=64, shuffle=True, collate_fn=None):
		self.dataset = dataset
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.collate_fn = collate_fn if collate_fn else self.collate_batch

	def __len__(self):
		return (len(self.dataset) + self.batch_size - 1) // self.batch_size

	def __iter__(self):
		indices = np.random.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))

		for i in range(0, len(indices), self.batch_size):
			batch_indices = indices[i: i+self.batch_size]
			batch = [self.dataset[k] for k in batch_indices]
			yield self.collate_fn(batch)

	def collate_batch(self, batch):
		x = [item[0] for item in batch]
		y = [item[1] for item in batch]
		return np.stack(x), np.stack(y)


def load_dataset(dataset_cls, data_dir='./data', split='train'):
	dataset = dataset_cls.__new__(dataset_cls)
	with open(os.path.join(data_dir, f'{split}.pkl'), 'rb') as f:
		dataset.load_state_dict(pickle.load(f))
	return dataset


def save_dataset(dataset, data_dir='./data', split='train'):
	os.makedirs(data_dir, exist_ok=True)
	with open(os.path.join(data_dir, f'{split}.pkl'), 'wb') as f:
		pickle.dump(dataset.state_dict(), f)
