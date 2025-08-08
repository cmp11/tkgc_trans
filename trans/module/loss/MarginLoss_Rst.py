import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss
'''
在原有loss的基础上，增加限制条件
'''
class MarginLoss_Rst(Loss):

	def __init__(self, adv_temperature = None, margin = 6.0, lam = 0.5):
		super(MarginLoss_Rst, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
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

	def forward(self, p_score, n_score, r_tp_cos):
		# print("r_tp_cos is : " , r_tp_cos)
		if self.adv_flag:
			return ((self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin)* (1-self.lam) + self.lam * (1-r_tp_cos)
		else:
			score = (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			# print("--------")
			# print("ori_score: ", score)
			# print("r_tp_cos: " , r_tp_cos)
			# print("lam: ", self.lam)
			# print("margin: ", self.margin)
			# print("--------")
			return score  + self.lam * (1-r_tp_cos)
			
	
	def predict(self, p_score, n_score,r_tp_cos):
		score = self.forward(p_score, n_score,r_tp_cos)
		return score.cpu().data.numpy()