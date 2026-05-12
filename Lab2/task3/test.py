import sys
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from task1.data_utils import load_data, CHINESE_ID2TAG, ENGLISH_ID2TAG
from task3.model import TransformerNER
from task3.data import NERDataset, collate_fn

if __name__ == "__main__":
	LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

	if LANG == "Chinese":
		id2tag = CHINESE_ID2TAG
		val_path = os.path.join("NER", "Chinese", "validation.txt")
		model_path = os.path.join("task3", "transformer_chinese.pt")
		output_path = os.path.join("task3", "pred_chinese.txt")
	else:
		id2tag = ENGLISH_ID2TAG
		val_path = os.path.join("NER", "English", "validation.txt")
		model_path = os.path.join("task3", "transformer_english.pt")
		output_path = os.path.join("task3", "pred_english.txt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(model_path, map_location=device, weights_only=False)
	tag2id = ckpt["tag2id"]
	bert_name = ckpt["bert_name"]
	tokenizer = BertTokenizer.from_pretrained(bert_name)

	sentences = load_data(val_path)
	dataset = NERDataset(sentences, tokenizer, tag2id)
	loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

	model = TransformerNER(bert_name, len(tag2id)).to(device)
	model.load_state_dict(ckpt["model"], strict=False)
	model.eval()

	all_preds = []
	with torch.no_grad():
		for input_ids, attention_mask, _, loss_mask in loader:
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			loss_mask = loss_mask.to(device)
			paths = model(input_ids, attention_mask, loss_mask=loss_mask)
			all_preds.append(paths[0].cpu().tolist())

	with open(output_path, "w", encoding="utf-8") as out:
		for (tokens, _), pred_ids in zip(sentences, all_preds):
			pos = 1
			for token in tokens:
				n = len(tokenizer.encode(token, add_special_tokens=False))
				tag_id = pred_ids[pos]
				out.write(f"{token} {id2tag[tag_id]}\n")
				pos += n
			out.write("\n")

	print(f"[{LANG}] prediction saved to {output_path}")

	from NER.check import check
	check(language=LANG, gold_path=val_path, my_path=output_path)
