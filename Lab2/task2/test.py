import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from task1.data_utils import load_data, CHINESE_ID2TAG, ENGLISH_ID2TAG
from task2.train import LinearCRF, NERDataset, collate_fn


LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

if LANG == "Chinese":
	id2tag = CHINESE_ID2TAG
	validation_path = os.path.join("NER", "Chinese", "validation.txt")
	model_path = os.path.join("task2", "crf_chinese.pt")
	output_path = os.path.join("task2", "pred_chinese.txt")
else:
	id2tag = ENGLISH_ID2TAG
	validation_path = os.path.join("NER", "English", "validation.txt")
	model_path = os.path.join("task2", "crf_english.pt")
	output_path = os.path.join("task2", "pred_english.txt")

ckpt = torch.load(model_path, weights_only=False)
vocab = ckpt["vocab"]
tag2id = ckpt["tag2id"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearCRF(len(vocab), len(tag2id)).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

sentences = load_data(validation_path)

dataset = NERDataset(sentences, vocab, tag2id)
loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

all_preds = []
with torch.no_grad():
	for input_ids, _, mask in loader:
		input_ids, mask = input_ids.to(device), mask.to(device)
		paths = model.decode(input_ids, mask)
		all_preds.extend(paths.cpu().numpy())

with open(output_path, "w", encoding="utf-8") as out:
	for (tokens, _), pred_tags in zip(sentences, all_preds):
		for token, tag_id in zip(tokens, pred_tags):
			out.write(f"{token} {id2tag[tag_id]}\n")
		out.write("\n")

print(f"[{LANG}] prediction saved to {output_path}")

from NER.check import check
check(language=LANG, gold_path=validation_path, my_path=output_path)
