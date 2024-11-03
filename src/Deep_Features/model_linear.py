import torch.nn as nn

class LinearPredictor(nn.Module):
	"""
	"""
	def __init__(self, in_dim, out_dim):
		super(LinearPredictor, self).__init__()
		self.in_dim = in_dim
		self.linear_op = nn.Linear(in_dim, out_dim)

	def forward(self, x):
		x = self.linear_op(x.reshape((-1, self.in_dim)))
		return x

def linear_predictor(in_dim, out_dim):
	return LinearPredictor(in_dim, out_dim)