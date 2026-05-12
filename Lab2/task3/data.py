import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils.data import load_data, CHINESE_TAG2ID, ENGLISH_TAG2ID


class NERDataset(Dataset):
	def __init__(self, sentences, tokenizer, tag2id, max_len=510):
		self.tokenizer = tokenizer
		self.tag2id = tag2id
		self.max_len = max_len
		self.samples = []
		for tokens, labels in sentences:
			sample = self._tokenize_and_align(tokens, labels)
			if sample is not None:
				self.samples.append(sample)

	def _tokenize_and_align(self, tokens, labels):
		token_ids = [self.tokenizer.cls_token_id]
		aligned_labels = [0]
		mask = [0]
		for t, l in zip(tokens, labels):
			subtokens = self.tokenizer.encode(t, add_special_tokens=False)
			if len(token_ids) + len(subtokens) + 1 > self.max_len:
				break
			tag_id = self.tag2id[l]
			for j, st in enumerate(subtokens):
				token_ids.append(st)
				aligned_labels.append(tag_id)
				mask.append(1 if j == 0 else 0)
		token_ids.append(self.tokenizer.sep_token_id)
		aligned_labels.append(0)
		mask.append(0)
		return token_ids, aligned_labels, mask

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		return self.samples[idx]


def collate_fn(batch):
	token_ids_list, labels_list, mask_list = zip(*batch)
	max_len = max(len(x) for x in token_ids_list)

	padded_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
	padded_labels = torch.full((len(batch), max_len), 0, dtype=torch.long)
	padded_mask = torch.zeros(len(batch), max_len, dtype=torch.float)
	attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

	for i in range(len(batch)):
		L = len(token_ids_list[i])
		padded_ids[i, :L] = torch.tensor(token_ids_list[i], dtype=torch.long)
		padded_labels[i, :L] = torch.tensor(labels_list[i], dtype=torch.long)
		padded_mask[i, :L] = torch.tensor(mask_list[i], dtype=torch.float)
		attention_mask[i, :L] = 1

	return padded_ids, attention_mask, padded_labels, padded_mask
