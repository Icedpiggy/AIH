def load_data(path):
	sentences = []
	tokens, labels = [], []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				if tokens:
					sentences.append((tokens, labels))
					tokens, labels = [], []
				continue
			parts = line.split()
			tokens.append(parts[0])
			labels.append(parts[1])
	if tokens:
		sentences.append((tokens, labels))
	return sentences


def build_vocab(sentences, unk_threshold=1):
	token_count = {}
	for tokens, _ in sentences:
		for t in tokens:
			token_count[t] = token_count.get(t, 0) + 1
	vocab = {"<UNK>": 0}
	for t, c in token_count.items():
		if c > unk_threshold:
			vocab[t] = len(vocab)
	return vocab


CHINESE_TAGS = [
	'O',
	'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME',
	'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT',
	'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU',
	'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE',
	'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG',
	'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE',
	'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO',
	'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC',
]

ENGLISH_TAGS = [
	'O',
	'B-PER', 'I-PER',
	'B-ORG', 'I-ORG',
	'B-LOC', 'I-LOC',
	'B-MISC', 'I-MISC',
]

CHINESE_TAG2ID = {t: i for i, t in enumerate(CHINESE_TAGS)}
CHINESE_ID2TAG = {i: t for i, t in enumerate(CHINESE_TAGS)}
ENGLISH_TAG2ID = {t: i for i, t in enumerate(ENGLISH_TAGS)}
ENGLISH_ID2TAG = {i: t for i, t in enumerate(ENGLISH_TAGS)}
