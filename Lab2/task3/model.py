import torch
import torch.nn as nn
from transformers import BertModel
from utils.crf import CRF

class TransformerNER(nn.Module):
	def __init__(self, bert_name, num_tags, unfreeze_layers=0):
		super().__init__()
		self.bert = BertModel.from_pretrained(bert_name)
		for p in self.bert.parameters():
			p.requires_grad = False
		if unfreeze_layers > 0:
			for layer in self.bert.encoder.layer[-unfreeze_layers:]:
				for p in layer.parameters():
					p.requires_grad = True
		self.dropout = nn.Dropout(0.1)
		self.fc = nn.Linear(self.bert.config.hidden_size, num_tags)
		self.crf = CRF(num_tags)

	def forward(self, input_ids, attention_mask, tags=None, loss_mask=None):
		hidden = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
		emissions = self.fc(self.dropout(hidden))
		if tags is not None:
			return self.crf.neg_log_likelihood(emissions, tags, loss_mask)
		else:
			return self.crf.decode(emissions, loss_mask)
