import torch
import torch.nn as nn


class CRF(nn.Module):
	def __init__(self, tag_size):
		super().__init__()
		self.tag_size = tag_size
		self.start_tr = nn.Parameter(torch.randn(tag_size))
		self.end_tr = nn.Parameter(torch.randn(tag_size))
		self.tr = nn.Parameter(torch.randn(tag_size, tag_size))

	def _forward_alg(self, emissions, mask):
		B, T, N = emissions.shape
		alpha = self.start_tr + emissions[:, 0, :]

		for i in range(1, T):
			emit = emissions[:, i, :]
			scores = alpha.unsqueeze(2) + self.tr.unsqueeze(0) + emit.unsqueeze(1)
			new_alpha = torch.logsumexp(scores, dim=1)

			m = mask[:, i].unsqueeze(1)
			alpha = new_alpha * m + alpha * (1 - m)

		alpha = alpha + self.end_tr
		return torch.logsumexp(alpha, dim=1)

	def _score_sentence(self, emissions, tags, mask):
		B, T, _ = emissions.shape

		emit_scores = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)
		emit_scores = (emit_scores * mask).sum(dim=1)

		trans_scores = torch.zeros(B, device=emissions.device)
		for i in range(T - 1):
			trans_scores += self.tr[tags[:, i], tags[:, i+1]] * mask[:, i+1]

		start_scores = self.start_tr[tags[:, 0]]

		lengths = mask.sum(dim=1).long()
		last_tags = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
		end_scores = self.end_tr[last_tags]

		return emit_scores + trans_scores + start_scores + end_scores

	def neg_log_likelihood(self, emissions, tags, mask):
		log_z = self._forward_alg(emissions, mask)
		gold_score = self._score_sentence(emissions, tags, mask)
		return (log_z - gold_score).mean()

	def decode(self, emissions, mask):
		B, T, N = emissions.shape
		scores = self.start_tr + emissions[:, 0, :]
		prev = torch.zeros((T, B, N), dtype=torch.int32, device=emissions.device)

		for i in range(1, T):
			s = scores.unsqueeze(2) + self.tr.unsqueeze(0) + emissions[:, i, :].unsqueeze(1)
			best_scores, prev[i, :, :] = s.max(dim=1)

			m = mask[:, i].unsqueeze(1)
			scores = best_scores * m + scores * (1 - m)

		scores = scores + self.end_tr
		best_last = scores.argmax(dim=1)

		paths = torch.zeros((B, T), dtype=torch.long, device=emissions.device)
		paths[:, T - 1] = best_last

		for i in range(T - 2, -1, -1):
			best_last = prev[i+1, :, :].gather(1, best_last.unsqueeze(1)).squeeze(1)
			paths[:, i] = best_last

		return paths
