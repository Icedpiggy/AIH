import sys
import os
import pickle
from task1.data_utils import load_data, build_vocab, CHINESE_TAG2ID, ENGLISH_TAG2ID
from task1.HMM import HMM

UNK = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

if LANG == "Chinese":
	tag2id = CHINESE_TAG2ID
	train_path = os.path.join("NER", "Chinese", "train.txt")
	save_path = os.path.join("task1", "hmm_chinese.pkl")
else:
	tag2id = ENGLISH_TAG2ID
	train_path = os.path.join("NER", "English", "train.txt")
	save_path = os.path.join("task1", "hmm_english.pkl")

sentences = load_data(train_path)
vocab = build_vocab(sentences, unk_threshold=UNK)

encoded = []
for tokens, labels in sentences:
	ids = [vocab.get(t, 0) for t in tokens]
	tag_ids = [tag2id[l] for l in labels]
	encoded.append((ids, tag_ids))

model = HMM(len(vocab), len(tag2id))
model.fit(encoded)

state = model.state_dict()
state["vocab"] = vocab
state["tag2id"] = tag2id
state["lang"] = LANG
with open(save_path, "wb") as f:
	pickle.dump(state, f)

print(f"[{LANG}] vocab={len(vocab)}, tags={len(tag2id)}, saved to {save_path}")
