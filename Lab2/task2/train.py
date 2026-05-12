import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data import load_data, build_vocab, CHINESE_TAG2ID, ENGLISH_TAG2ID
from task2.model import LinearCRF
from task2.data import NERDataset, collate_fn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
	LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

	if LANG == "Chinese":
		tag2id = CHINESE_TAG2ID
		train_path = os.path.join("NER", "Chinese", "train.txt")
		save_path = os.path.join("task2", "crf_chinese.pt")
		# Best params from experiment: hidden_dim=64, unk_threshold=1, lr=3e-3
		best_hidden_dim = 64
		best_unk_threshold = 1
		best_lr = 3e-3
	else:
		tag2id = ENGLISH_TAG2ID
		train_path = os.path.join("NER", "English", "train.txt")
		save_path = os.path.join("task2", "crf_english.pt")
		# Best params from experiment: hidden_dim=256, unk_threshold=0, lr=3e-3
		best_hidden_dim = 256
		best_unk_threshold = 0
		best_lr = 3e-3

	sentences = load_data(train_path)
	vocab = build_vocab(sentences, unk_threshold=best_unk_threshold)

	dataset = NERDataset(sentences, vocab, tag2id)
	loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = LinearCRF(len(vocab), len(tag2id), hidden_dim=best_hidden_dim).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)

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
		"hidden_dim": best_hidden_dim,
	}, save_path)
	print(f"[{LANG}] saved to {save_path}")
