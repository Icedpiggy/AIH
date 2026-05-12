import sys
import os
import pickle
from utils.data import load_data, CHINESE_TAGS, ENGLISH_TAGS, CHINESE_ID2TAG, ENGLISH_ID2TAG
from task1.HMM import HMM

LANG = sys.argv[1] if len(sys.argv) > 1 else "Chinese"

if LANG == "Chinese":
	id2tag = CHINESE_ID2TAG
	validation_path = os.path.join("NER", "Chinese", "validation.txt")
	model_path = os.path.join("task1", "hmm_chinese.pkl")
	output_path = os.path.join("task1", "pred_chinese.txt")
else:
	id2tag = ENGLISH_ID2TAG
	validation_path = os.path.join("NER", "English", "validation.txt")
	model_path = os.path.join("task1", "hmm_english.pkl")
	output_path = os.path.join("task1", "pred_english.txt")

with open(model_path, "rb") as f:
	state = pickle.load(f)

vocab = state["vocab"]
model = HMM.from_state_dict(state)

sentences = load_data(validation_path)

encoded = []
for tokens, _ in sentences:
	encoded.append([vocab.get(t, 0) for t in tokens])

preds = model(encoded)

with open(output_path, "w", encoding="utf-8") as out:
	for (tokens, _), pred_tags in zip(sentences, preds):
		for token, tag_id in zip(tokens, pred_tags):
			out.write(f"{token} {id2tag[tag_id]}\n")
		out.write("\n")

print(f"[{LANG}] prediction saved to {output_path}")

from NER.check import check
check(language=LANG, gold_path=validation_path, my_path=output_path)
