import numpy as np

class Dataset:
	def __init__(self):
		pass

	def __len__(self):
		raise NotImplementedError(f"{self.__class__.__name__}.__len__() not implemented")

	def __getitem__(self, idx):
		raise NotImplementedError(f"{self.__class__.__name__}.__getitem__() not implemented")


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
		return np.stack(x), np.array(y)