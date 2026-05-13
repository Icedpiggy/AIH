import sys
import os
import torch
from torch.utils.data import DataLoader
from utils.data import load_data, CHINESE_ID2TAG, ENGLISH_ID2TAG
from extra.model import TemplateCRF
from extra.template import parse_template, build_feature_vocab
from extra.train import featurize_sentence, collate_fn

TEMPLATE_PATH = os.path.join("NER", "template_for_crf.utf8")


if __name__ == "__main__":
	LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

	if LANG == "Chinese":
		id2tag = CHINESE_ID2TAG
		val_path = os.path.join("NER", "Chinese", "validation.txt")
		model_path = os.path.join("extra", "template_chinese.pt")
		output_path = os.path.join("extra", "pred_chinese.txt")
	else:
		id2tag = ENGLISH_ID2TAG
		val_path = os.path.join("NER", "English", "validation.txt")
		model_path = os.path.join("extra", "template_english.pt")
		output_path = os.path.join("extra", "pred_english.txt")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	ckpt = torch.load(model_path, map_location=device, weights_only=False)
	tag2id = ckpt["tag2id"]
	uni_vocab = ckpt["uni_vocab"]
	bi_vocab = ckpt["bi_vocab"]

	sentences = load_data(val_path)
	unigrams, bigrams = parse_template(TEMPLATE_PATH)
	featurized = [featurize_sentence(s, unigrams, bigrams, uni_vocab, bi_vocab) for s in sentences]
	dataset = [(u, b, [tag2id[l] for l in s[1]], s[0]) for (u, b), s in zip(featurized, sentences)]
	loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

	model = TemplateCRF(len(uni_vocab), len(bi_vocab), len(tag2id)).to(device)
	model.load_state_dict(ckpt["model"])
	model.eval()

	all_preds = []
	with torch.no_grad():
		for uni_ids, bi_ids, _, mask, tokens in loader:
			uni_ids = uni_ids.to(device)
			bi_ids = bi_ids.to(device)
			mask = mask.to(device)
			paths = model.decode(uni_ids, bi_ids, mask)
			all_preds.extend(zip(tokens, paths.cpu().tolist()))

	with open(output_path, "w", encoding="utf-8") as out:
		for tokens, pred_ids in all_preds:
			L = len(tokens)
			for i in range(L):
				out.write(f"{tokens[i]} {id2tag[pred_ids[i]]}\n")
			out.write("\n")

	print(f"[{LANG}] prediction saved to {output_path}")

	from NER.check import check
	check(language=LANG, gold_path=val_path, my_path=output_path)
