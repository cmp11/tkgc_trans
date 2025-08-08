from .Strategy import Strategy

class NegativeSampling_Con(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0):
		super(NegativeSampling_Con, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate

	def _get_positive_score(self, score):
		positive_score = score[:self.batch_size]
		positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score):
		negative_score = score[self.batch_size:]
		negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
		return negative_score

	def forward(self, data):
		score, r_tp_cos_score = self.model(data)
		p_score = self._get_positive_score(score)
		n_score = self._get_negative_score(score)

		cos_p_score = None
		cos_n_score = None
		if r_tp_cos_score is not None:
			cos_p_score = self._get_positive_score(r_tp_cos_score)
			cos_n_score = self._get_negative_score(r_tp_cos_score)
		loss_res = self.loss(p_score, n_score, cos_p_score, cos_n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res