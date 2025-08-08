import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model
'''
transR的算法模型，将实体和关系映射到关系的空间维度,映射到关系空间后，增加限制条件，关系和时间的embedding方向相同。
'''
class TransR_AddTime_r_Rst(Model):
    

    def __init__(self, ent_tot, rel_tot, tp_tot,dim_e = 100, dim_r = 100, dim_tp = 100,  p_norm = 1, norm_flag = True, rand_init = False, margin = None):
        super(TransR_AddTime_r_Rst, self).__init__(ent_tot, rel_tot,tp_tot)
        
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

        self.e_to_r_transfer_matrix = nn.Embedding(self.rel_tot, self.dim_e * self.dim_r)
        self.tp_to_r_transfer_matrix = nn.Embedding(self.rel_tot, self.dim_tp * self.dim_r)
  
        if not self.rand_init:
            identity = torch.zeros(self.dim_e, self.dim_r)
            for i in range(min(self.dim_e, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_e)
            for i in range(self.rel_tot):
                self.e_to_r_transfer_matrix.weight.data[i] = identity

            identity = torch.zeros(self.dim_tp, self.dim_r)
            for i in range(min(self.dim_tp, self.dim_r)):
                identity[i][i] = 1
            identity = identity.view(self.dim_r * self.dim_tp)
            for i in range(self.rel_tot):
                self.tp_to_r_transfer_matrix.weight.data[i] = identity
    
        else:
            nn.init.xavier_uniform_(self.e_to_r_transfer_matrix.weight.data)
            nn.init.xavier_uniform_(self.tp_to_r_transfer_matrix.weight.data)

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
    
    def _e_transfer(self, e, e_to_r_transfer):
        e_to_r_transfer = e_to_r_transfer.view(-1, self.dim_e, self.dim_r)
        
        if e.shape[0] != e_to_r_transfer.shape[0]:
            e = e.view(-1, e_to_r_transfer.shape[0], self.dim_e).permute(1, 0, 2)
            e = torch.matmul(e, e_to_r_transfer).permute(1, 0, 2)
        else:
            e = e.view(-1, 1, self.dim_e)
            e = torch.matmul(e, e_to_r_transfer)
        return e.view(-1, self.dim_r)

    def _tp_transfer(self, tp, tp_to_r_transfer):
        tp_to_r_transfer = tp_to_r_transfer.view(-1, self.dim_tp, self.dim_r)
        if tp.shape[0] != tp_to_r_transfer.shape[0]:
            tp = tp.view(-1, tp_to_r_transfer.shape[0], self.dim_tp).permute(1, 0, 2)
            tp = torch.matmul(tp, tp_to_r_transfer).permute(1, 0, 2)
        else:
            tp = tp.view(-1, 1, self.dim_tp)
            tp = torch.matmul(tp, tp_to_r_transfer)
        return tp.view(-1, self.dim_r)

    def cosine_similarity_matrix(self,A, B):
        # 计算向量的范数（长度）
        norm_A = torch.norm(A, dim=1, keepdim=True)
        norm_B = torch.norm(B, dim=1, keepdim=True)
        # print("norm_A: ",norm_A)
        # print("norm_B: ",norm_B)
        # 计算点积
        dot_product = torch.sum(A * B, dim=1, keepdims=True)
        # print("dot_product: ",dot_product)
        
        # 计算余弦相似度
        # print("d",norm_A * norm_B)
        similarity_matrix = dot_product / (norm_A * norm_B)
        # similarity_mean = torch.abs(similarity_matrix.flatten()).mean()
        similarity_mean = similarity_matrix.flatten().mean()
        return similarity_mean
 
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
        e_to_r_transfer = self.e_to_r_transfer_matrix(batch_r)
        tp_to_r_transfer = self.tp_to_r_transfer_matrix(batch_r)
        h = self._e_transfer(h, e_to_r_transfer)
        t = self._e_transfer(t, e_to_r_transfer)
        tp = self._tp_transfer(tp, tp_to_r_transfer)
        score = self._calc(h ,t, r, tp,mode)
  
        # 计算关系空间上 关系和时间的方向差距
        # r_embedding = r[:batch_size, :]
        # tp_embedding = tp[:batch_size, :]
        r_tp_cos_score = self.cosine_similarity_matrix(r,tp)
        # print("r_tp_cos_score",r_tp_cos_score)

        if self.margin_flag:
            return self.margin - score,r_tp_cos_score
        else:
            return score,r_tp_cos_score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        batch_tp = data['batch_tp']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        tp = self.tp_embeddings(batch_tp)
  
        e_to_r_transfer = self.e_to_r_transfer_matrix(batch_r)
        tp_to_r_transfer = self.tp_to_r_transfer_matrix(batch_r)
  
        regul = (torch.mean(h ** 2) + 
                 torch.mean(t ** 2) + 
                 torch.mean(r ** 2) +
                 torch.mean(tp ** 2) +
                 torch.mean(e_to_r_transfer ** 2)+
                  torch.mean(tp_to_r_transfer ** 2)) / 6
        return regul * regul

    def predict(self, data):
        score,r_tp_cos_score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy(),r_tp_cos_score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy(),r_tp_cos_score.cpu().data.numpy()