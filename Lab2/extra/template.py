import os


def parse_template(template_path):
	unigrams = []
	bigrams = []
	with open(template_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split(":", 1)
			if len(parts) != 2:
				continue
			pattern_body = parts[1].strip()
			offsets = []
			for segment in pattern_body.split("/"):
				segment = segment.strip()
				if segment.startswith("%x[") and segment.endswith("]"):
					inner = segment[3:-1]
					row_str, col_str = inner.split(",")
					offsets.append((int(row_str.strip()), int(col_str.strip())))
			if line.startswith("U"):
				unigrams.append(offsets)
			elif line.startswith("B"):
				bigrams.append(offsets)
	return unigrams, bigrams


def extract_features(sentence, unigrams, bigrams, position):
	tokens = sentence[0]
	T = len(tokens)
	uni_feats = []
	for ut in unigrams:
		vals = []
		for row, col in ut:
			idx = position + row
			if 0 <= idx < T:
				vals.append(tokens[idx])
			else:
				vals.append("_BOS_" if idx < 0 else "_EOS_")
		uni_feats.append("/".join(vals))
	bi_feats = []
	for bt in bigrams:
		vals = []
		for row, col in bt:
			idx = position + row
			if 0 <= idx < T:
				vals.append(tokens[idx])
			else:
				vals.append("_BOS_" if idx < 0 else "_EOS_")
		bi_feats.append("/".join(vals))
	return uni_feats, bi_feats


def build_feature_vocab(sentences, unigrams, bigrams, min_count=1):
	uni_freq = {}
	bi_freq = {}
	for sent in sentences:
		T = len(sent[0])
		for pos in range(T):
			uf, bf = extract_features(sent, unigrams, bigrams, pos)
			for f in uf:
				uni_freq[f] = uni_freq.get(f, 0) + 1
			for f in bf:
				bi_freq[f] = bi_freq.get(f, 0) + 1
	uni_vocab = {}
	bi_vocab = {}
	for f, c in uni_freq.items():
		if c >= min_count:
			uni_vocab[f] = len(uni_vocab)
	for f, c in bi_freq.items():
		if c >= min_count:
			bi_vocab[f] = len(bi_vocab)
	return uni_vocab, bi_vocab
