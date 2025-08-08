import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
'''
transR的算法模型，将实体和关系映射到时间的空间维度
映射时，使用共享矩阵 即所有的实体用同一个矩阵进行映射，所有的关系用同一个矩阵映射
'''
class TransR_tp_AddTime(Model):
    

    def __init__(self, ent_tot, rel_tot, tp_tot,dim_e = 100, dim_r = 100, dim_tp = 100,  p_norm = 1, norm_flag = True, rand_init = False, margin = None):
        super(TransR_tp_AddTime, self).__init__(ent_tot, rel_tot,tp_tot)
        
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.dim_tp = dim_tp
  
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.rand_init = rand_init
  


        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.tp_embeddings = nn.Embedding(self.tp_tot, self.dim_tp)
  
  
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
        nn.init.xavier_uniform_(self.tp_embeddings.weight.data)
  
        self.e_to_tp_transfer_matrix = nn.Embedding(self.dim_e , self.dim_tp)
        self.r_to_tp_transfer_matrix = nn.Embedding(self.dim_r , self.dim_tp)
  
        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_tp)
            for i in range(min(self.dim_e, self.dim_tp)):
                identity[i][i] = 1
            self.e_to_tp_transfer_matrix.weight.data.copy_(identity)
            

            identity = torch.zeros(self.dim_r, self.dim_tp)
            for i in range(min(self.dim_r , self.dim_tp)):
                identity[i][i] = 1
            self.r_to_tp_transfer_matrix.weight.data.copy_(identity)
    
        else:
            nn.init.xavier_uniform_(self.e_to_tp_transfer_matrix.weight.data)
            nn.init.xavier_uniform_(self.r_to_tp_transfer_matrix.weight.data)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _calc(self, h, t, r,tp, mode):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
            tp = F.normalize(tp, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
            tp = tp.view(-1, tp.shape[0], tp.shape[-1])
        if mode == 'head_batch':
            # score = h + (r - t)
            score = h + (r + tp- t)
               
        else:
            # score = (h + r) - t
            score = (h + r + tp) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score
    
    def _e_transfer(self, e):
        # e: [batch_size, dim_e]
        # weight: [dim_e, dim_tp]
        transfer_matrix = self.e_to_tp_transfer_matrix.weight  # [dim_e, dim_tp]
        return torch.matmul(e, transfer_matrix)   # [batch_size, dim_tp]
       

    def _r_transfer(self, r):
        transfer_matrix = self.r_to_tp_transfer_matrix.weight  # [dim_r, dim_tp]
        return torch.matmul(r, transfer_matrix)  # [batch_size, dim_tp]

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        batch_tp = data['batch_tp']
        
        mode = data['mode']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        tp = self.tp_embeddings(batch_tp)
       
        h = self._e_transfer(h)
        t = self._e_transfer(t)
        r = self._r_transfer(r)
        score = self._calc(h ,t, r, tp,mode)
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        batch_tp = data['batch_tp']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        tp = self.tp_embeddings(batch_tp)
  
        
        e_to_tp_transfer = self.e_to_tp_transfer_matrix.weight
        r_to_tp_transfer = self.r_to_tp_transfer_matrix.weight
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2) +
                 torch.mean(tp ** 2) +
                 torch.mean(e_to_tp_transfer ** 2)+
                  torch.mean(r_to_tp_transfer ** 2)) / 6
        return regul * regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()