import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from task1.data_utils import load_data, build_vocab, CHINESE_TAG2ID, ENGLISH_TAG2ID
from task2.CRF import CRF


class LinearCRF(nn.Module):
	def __init__(self, vocab_size, num_tags, hidden_dim=128):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, hidden_dim)
		self.fc = nn.Linear(hidden_dim, num_tags)
		self.crf = CRF(num_tags)

	def forward(self, input_ids, tags, mask):
		emi = self.fc(self.embed(input_ids))
		return self.crf.neg_log_likelihood(emi, tags, mask)

	def decode(self, input_ids, mask):
		emi = self.fc(self.embed(input_ids))
		return self.crf.decode(emi, mask)


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


if __name__ == "__main__":
	LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

	if LANG == "Chinese":
		tag2id = CHINESE_TAG2ID
		train_path = os.path.join("NER", "Chinese", "train.txt")
		save_path = os.path.join("task2", "crf_chinese.pt")
	else:
		tag2id = ENGLISH_TAG2ID
		train_path = os.path.join("NER", "English", "train.txt")
		save_path = os.path.join("task2", "crf_english.pt")

	sentences = load_data(train_path)
	vocab = build_vocab(sentences, unk_threshold=0)

	dataset = NERDataset(sentences, vocab, tag2id)
	loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = LinearCRF(len(vocab), len(tag2id)).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	for epoch in range(30):
		total_loss = 0
		for input_ids, tags, mask in loader:
			input_ids, tags, mask = input_ids.to(device), tags.to(device), mask.to(device)
			loss = model(input_ids, tags, mask)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch {epoch + 1}/30  loss={total_loss / len(loader):.4f}")

	torch.save({
		"model": model.state_dict(),
		"vocab": vocab,
		"tag2id": tag2id,
		"lang": LANG,
	}, save_path)
	print(f"[{LANG}] saved to {save_path}")
