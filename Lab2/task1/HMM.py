import numpy as np

class HMM:
	def __init__(self, vocab_size, tag_size):
		self.vocab_size = vocab_size
		self.tag_size = tag_size
		self.pi = None
		self.A = None
		self.B = None

	def fit(self, data):
		t0 = np.zeros(self.tag_size)
		t1 = np.zeros(self.tag_size)
		tt = np.zeros((self.tag_size, self.tag_size))
		tv = np.zeros((self.tag_size, self.vocab_size))

		for x, y in data:
			t0[y[0]] += 1
			t1[y[1:]] += 1
			tt[y[:-1], y[1:]] += 1
			tv[y, x] += 1
		
		self.pi = np.log(t0 + 1) - np.log(len(data) + self.tag_size)
		self.A = np.log(tt + 1) - np.log(t1 + self.tag_size).reshape(-1, 1)
		self.B = np.log(tv + 1) - np.log(t0 + t1 + self.vocab_size).reshape(-1, 1)

	def forward(self, data):
		result = []
		for x in data:
			V = np.zeros(self.tag_size)
			prev = np.zeros((len(x), self.tag_size), dtype=np.int32)
			V = self.pi + self.B[:, x[0]]
			prev[0, :] = -1
			for i in range(1, len(x)):
				score = V.reshape(-1, 1) + self.A + self.B[:, x[i]].reshape(1, -1)
				V, prev[i, :] = np.max(score, axis=0), np.argmax(score, axis=0)
			
			y = []
			cur = np.argmax(V)
			for i in range(len(x)-1, -1, -1):
				y.append(cur)
				cur = prev[i, cur]

			y.reverse()
			result.append(y.copy())
		return result

	def state_dict(self):
		return {'pi': self.pi, 'A': self.A, 'B': self.B, 'vocab_size': self.vocab_size, 'tag_size': self.tag_size}

	def load_state_dict(self, data):
		self.vocab_size = data['vocab_size']
		self.tag_size = data['tag_size']
		self.pi = data['pi']
		self.A = data['A']
		self.B = data['B']

	@classmethod
	def from_state_dict(cls, data):
		model = cls(data['vocab_size'], data['tag_size'])
		model.load_state_dict(data)
		return model

	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)