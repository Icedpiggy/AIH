import sys
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.data import load_data, CHINESE_TAG2ID, ENGLISH_TAG2ID
from extra.model import TemplateCRF
from extra.template import parse_template, extract_features, build_feature_vocab

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TEMPLATE_PATH = os.path.join("NER", "template_for_crf.utf8")


def featurize_sentence(sentence, unigrams, bigrams, uni_vocab, bi_vocab):
	tokens = sentence[0]
	T = len(tokens)
	Nu = len(unigrams)
	Nb = len(bigrams)
	uni_ids = torch.full((T, Nu), -1, dtype=torch.long)
	bi_ids = torch.full((T, Nb), -1, dtype=torch.long)
	for pos in range(T):
		uf, bf = extract_features(sentence, unigrams, bigrams, pos)
		for j, f in enumerate(uf):
			if f in uni_vocab:
				uni_ids[pos, j] = uni_vocab[f]
		for j, f in enumerate(bf):
			if f in bi_vocab:
				bi_ids[pos, j] = bi_vocab[f]
	return uni_ids, bi_ids


def collate_fn(batch):
	uni_list, bi_list, tags_list = zip(*batch)
	lengths = [x.shape[0] for x in uni_list]
	max_len = max(lengths)
	Nu = uni_list[0].shape[1]
	Nb = bi_list[0].shape[1]

	padded_uni = torch.full((len(batch), max_len, Nu), -1, dtype=torch.long)
	padded_bi = torch.full((len(batch), max_len, Nb), -1, dtype=torch.long)
	padded_tags = torch.zeros(len(batch), max_len, dtype=torch.long)
	padded_mask = torch.zeros(len(batch), max_len, dtype=torch.float)

	for i in range(len(batch)):
		L = lengths[i]
		padded_uni[i, :L] = uni_list[i]
		padded_bi[i, :L] = bi_list[i]
		padded_tags[i, :L] = torch.tensor(tags_list[i], dtype=torch.long)
		padded_mask[i, :L] = 1.0

	return padded_uni, padded_bi, padded_tags, padded_mask


if __name__ == "__main__":
	LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

	if LANG == "Chinese":
		tag2id = CHINESE_TAG2ID
		train_path = os.path.join("NER", "Chinese", "train.txt")
		save_path = os.path.join("extra", "template_chinese.pt")
	else:
		tag2id = ENGLISH_TAG2ID
		train_path = os.path.join("NER", "English", "train.txt")
		save_path = os.path.join("extra", "template_english.pt")

	sentences = load_data(train_path)
	unigrams, bigrams = parse_template(TEMPLATE_PATH)
	uni_vocab, bi_vocab = build_feature_vocab(sentences, unigrams, bigrams)

	featurized = [featurize_sentence(s, unigrams, bigrams, uni_vocab, bi_vocab) for s in sentences]
	dataset = [(u, b, [tag2id[l] for l in s[1]]) for (u, b), s in zip(featurized, sentences)]
	loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	print(f"Uni features: {len(uni_vocab)}, Bi features: {len(bi_vocab)}")

	model = TemplateCRF(len(uni_vocab), len(bi_vocab), len(tag2id)).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

	for epoch in range(30):
		total_loss = 0
		for uni_ids, bi_ids, tags, mask in loader:
			uni_ids = uni_ids.to(device)
			bi_ids = bi_ids.to(device)
			tags = tags.to(device)
			mask = mask.to(device)
			loss = model(uni_ids, bi_ids, tags, mask)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch {epoch + 1}/30  loss={total_loss / len(loader):.4f}")

	torch.save({
		"model": model.state_dict(),
		"uni_vocab": uni_vocab,
		"bi_vocab": bi_vocab,
		"tag2id": tag2id,
		"lang": LANG,
	}, save_path)
	print(f"[{LANG}] saved to {save_path}")
