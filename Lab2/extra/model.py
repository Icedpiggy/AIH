import torch
import torch.nn as nn


class TemplateCRF(nn.Module):
	def __init__(self, num_uni_features, num_bi_features, num_tags):
		super().__init__()
		self.num_tags = num_tags
		self.start_tr = nn.Parameter(torch.randn(num_tags))
		self.end_tr = nn.Parameter(torch.randn(num_tags))
		self.base_tr = nn.Parameter(torch.randn(num_tags, num_tags))
		self.uni_weights = nn.Parameter(torch.zeros(num_uni_features, num_tags))
		self.bi_weights = nn.Parameter(torch.zeros(num_bi_features, num_tags, num_tags))

	def _compute_emissions_and_tr_adj(self, batch_uni_ids, batch_bi_ids, batch_mask):
		B, T, U = batch_uni_ids.shape
		_, _, Bf = batch_bi_ids.shape
		N = self.num_tags

		emissions = torch.zeros(B, T, N, device=batch_uni_ids.device)
		for k in range(U):
			fids = batch_uni_ids[:, :, k]
			valid = fids >= 0
			w = self.uni_weights[fids.clamp(min=0)]
			w[~valid] = 0.0
			emissions = emissions + w

		tr_adj = torch.zeros(B, T, N, N, device=batch_bi_ids.device)
		for k in range(Bf):
			fids = batch_bi_ids[:, :, k]
			valid = fids >= 0
			w = self.bi_weights[fids.clamp(min=0)]
			w[~valid] = 0.0
			tr_adj = tr_adj + w

		return emissions, tr_adj

	def _forward_alg(self, emissions, tr_adj, mask):
		B, T, N = emissions.shape
		alpha = self.start_tr + emissions[:, 0, :]

		for i in range(1, T):
			tr = self.base_tr + tr_adj[:, i, :, :]
			emit = emissions[:, i, :]
			scores = alpha.unsqueeze(2) + tr + emit.unsqueeze(1)
			new_alpha = torch.logsumexp(scores, dim=1)
			m = mask[:, i].unsqueeze(1)
			alpha = new_alpha * m + alpha * (1 - m)

		alpha = alpha + self.end_tr
		return torch.logsumexp(alpha, dim=1)

	def _score_sentence(self, emissions, tr_adj, tags, mask):
		B, T, _ = emissions.shape

		emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
		emit_scores = (emit_scores * mask).sum(dim=1)

		trans_scores = torch.zeros(B, device=emissions.device)
		for i in range(T - 1):
			tr = self.base_tr + tr_adj[:, i, :, :]
			trans_scores += tr[torch.arange(B, device=emissions.device), tags[:, i], tags[:, i+1]] * mask[:, i+1]

		start_scores = self.start_tr[tags[:, 0]]
		lengths = mask.sum(dim=1).long()
		last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
		end_scores = self.end_tr[last_tags]

		return emit_scores + trans_scores + start_scores + end_scores

	def forward(self, batch_uni_ids, batch_bi_ids, tags, mask):
		emissions, tr_adj = self._compute_emissions_and_tr_adj(batch_uni_ids, batch_bi_ids, mask)
		log_z = self._forward_alg(emissions, tr_adj, mask)
		gold_score = self._score_sentence(emissions, tr_adj, tags, mask)
		return (log_z - gold_score).mean()

	def decode(self, batch_uni_ids, batch_bi_ids, mask):
		emissions, tr_adj = self._compute_emissions_and_tr_adj(batch_uni_ids, batch_bi_ids, mask)
		B, T, N = emissions.shape
		scores = self.start_tr + emissions[:, 0, :]
		prev = torch.zeros((T, B, N), dtype=torch.int32, device=emissions.device)

		for i in range(1, T):
			tr = self.base_tr + tr_adj[:, i, :, :]
			s = scores.unsqueeze(2) + tr + emissions[:, i, :].unsqueeze(1)
			best_scores, prev[i, :, :] = s.max(dim=1)
			m = mask[:, i].unsqueeze(1)
			scores = best_scores * m + scores * (1 - m)

		scores = scores + self.end_tr
		best_last = scores.argmax(dim=1)

		paths = torch.zeros((B, T), dtype=torch.long, device=emissions.device)
		paths[:, T-1] = best_last
		for i in range(T - 2, -1, -1):
			best_last = prev[i+1, :, :].gather(1, best_last.unsqueeze(1)).squeeze(1)
			paths[:, i] = best_last
		return paths
