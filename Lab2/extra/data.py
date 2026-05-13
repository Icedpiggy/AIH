import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
	def __init__(self, sentences, vocab, tag2id):
		self.sentences = sentences
		self.vocab = vocab
		self.tag2id = tag2id

	def __len__(self):
		return len(self.sentences)

	def __getitem__(self, idx):
		tokens, labels = self.sentences[idx]
		ids = [self.vocab.get(t, 0) for t in tokens]
		tags = [self.tag2id[l] for l in labels]
		return torch.tensor(ids, dtype=torch.long), torch.tensor(tags, dtype=torch.long)

def collate_fn(batch):
	ids_list, tags_list = zip(*batch)
	lengths = [len(x) for x in ids_list]
	max_len = max(lengths)

	padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
	padded_tags = torch.zeros(len(batch), max_len, dtype=torch.long)
	mask = torch.zeros(len(batch), max_len, dtype=torch.float)

	for i, (ids, tags) in enumerate(zip(ids_list, tags_list)):
		padded_ids[i, :len(ids)] = ids
		padded_tags[i, :len(tags)] = tags
		mask[i, :len(ids)] = 1.0

	return padded_ids, padded_tags, mask
