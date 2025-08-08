import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss
'''
在原有loss的基础上，增加限制条件
'''
class MarginLoss_Con(Loss):

	def __init__(self, adv_temperature = None, margin = 6.0, margin_con = 6.0, lam = 0.1):
		super(MarginLoss_Con, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin_con = nn.Parameter(torch.Tensor([margin_con]))
		self.lam = nn.Parameter(torch.Tensor([lam]))
		self.margin.requires_grad = False
		self.lam.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score, cos_p_score, cos_n_score):
		# if self.adv_flag:
		# 	return ((self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin)* (1-self.lam) + self.lam * (1-r_tp_cos)
		# else:
		score = (torch.max(p_score - n_score, -self.margin)).mean() + self.margin

		if cos_p_score is None or cos_n_score is None:
			r_tp_cos_score = 0.0
		else:
			r_tp_cos_score = (torch.max(cos_n_score - cos_p_score, -self.margin_con)).mean() + self.margin_con
		# print(f"score: {score}")
		# print(f"r_tp_cos_score: {r_tp_cos_score}")

		return score  + self.lam * r_tp_cos_score
			
	
	def predict(self, p_score, n_score, cos_p_score, cos_n_score):
		score = self.forward(p_score, n_score, cos_p_score, cos_n_score)
		return score.cpu().data.numpy()