import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from utils.data import load_data, CHINESE_TAG2ID, ENGLISH_TAG2ID
from task3.model import TransformerNER
from task3.data import NERDataset, collate_fn

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
		save_path = os.path.join("task3", "transformer_chinese.pt")
		bert_name = "bert-base-chinese"
	else:
		tag2id = ENGLISH_TAG2ID
		train_path = os.path.join("NER", "English", "train.txt")
		save_path = os.path.join("task3", "transformer_english.pt")
		bert_name = "bert-base-uncased"

	sentences = load_data(train_path)
	tokenizer = BertTokenizer.from_pretrained(bert_name)

	dataset = NERDataset(sentences, tokenizer, tag2id)
	loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	best_lr = 5e-5
	best_unfreeze = 7 if LANG == "Chinese" else 8
	model = TransformerNER(bert_name, len(tag2id), unfreeze_layers=best_unfreeze).to(device)

	trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Trainable params: {trainable}")

	optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)

	for epoch in range(5):
		total_loss = 0
		for input_ids, attention_mask, tags, loss_mask in loader:
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			tags = tags.to(device)
			loss_mask = loss_mask.to(device)
			loss = model(input_ids, attention_mask, tags=tags, loss_mask=loss_mask)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch {epoch + 1}/5  loss={total_loss / len(loader):.4f}")

	torch.save({
		"model": model.state_dict(),
		"tag2id": tag2id,
		"lang": LANG,
		"bert_name": bert_name,
	}, save_path)
	print(f"[{LANG}] saved to {save_path}")
