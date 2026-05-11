import torch.nn as nn
from task2.CRF import CRF


class LinearCRF(nn.Module):
	def __init__(self, vocab_size, num_tags, hidden_dim=128):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, hidden_dim)
		self.fc = nn.Linear(hidden_dim, num_tags)
		self.crf = CRF(num_tags)

	def forward(self, input_ids, tags, mask):
		emi = self.fc(self.embed(input_ids))
		return self.crf.neg_log_likelihood(emi, tags, mask)

	def decode(self, input_ids, mask):
		emi = self.fc(self.embed(input_ids))
		return self.crf.decode(emi, mask)
