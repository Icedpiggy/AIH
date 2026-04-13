from utils.backend import np
from .core import *

def xavier_uniform(input_dim, output_dim, gain=1.0):
	limit = gain * np.sqrt(6.0 / (input_dim + output_dim))
	return np.random.uniform(-limit, limit, size=(input_dim, output_dim))

def he_normal(input_dim, output_dim, gain=1.0):
	std = gain * np.sqrt(2.0 / input_dim)
	return np.random.randn(input_dim, output_dim) * std

def random_array(input_dim, output_dim, gain=1.0, random_policy=None):
	if random_policy == 'Xavier':
		return xavier_uniform(input_dim, output_dim, gain)
	elif random_policy == 'He':
		return he_normal(input_dim, output_dim, gain)
	else:
		return np.random.randn(input_dim, output_dim) * gain