# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
from sklearn.metrics import roc_auc_score
import copy
from tqdm import tqdm

class Tester_AddTime(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True):

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.rel_time_constrain = {}
        
        
        if self.use_gpu:
            self.model.cuda()
            
        

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def test_one_step(self, data):        
        return self.model.predict({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_tp': self.to_var(data['batch_tp'], self.use_gpu),
            'is_test': True,
            'mode': data['mode']
        })
        
        
    def hit_n(self,rank, n_hit):
        hit_n_num = len(list(filter(lambda x: x <= n_hit, rank)))
        return "{:.6f}".format(hit_n_num/len(rank))

    
    def get_mean_rank(self,rank):
        return "{:.6f}".format(sum(rank)/len(rank))
    
    def get_mean_reciprocal_rank(self,rank):
        mrr = sum(1 / x for x in rank) / len(rank)
        return "{:.6f}".format(mrr)
    
    def print_result(self,raw_array,raw_name):
        
        # 输出数据
        row = "{:<20}".format(raw_name)
        for m in raw_array:
            row += "{:<20}".format(m)
        print(row)
    
    def calc(self,head_rank,tail_rank):
        
        l_raw = []
        r_raw = []
        averaged_raw = []
        
        r_raw.append(self.get_mean_reciprocal_rank(tail_rank))
        r_raw.append(self.get_mean_rank(tail_rank))
        r_raw.append(self.hit_n(tail_rank,1))
        r_raw.append(self.hit_n(tail_rank,3))
        r_raw.append(self.hit_n(tail_rank,10))
        
        l_raw.append(self.get_mean_reciprocal_rank(head_rank))
        l_raw.append(self.get_mean_rank(head_rank))
        l_raw.append(self.hit_n(head_rank,1))
        l_raw.append(self.hit_n(head_rank,3))
        l_raw.append(self.hit_n(head_rank,10))
        
        averaged_raw = [((float)(x) + (float)(y)) / 2 for x, y in zip(l_raw, r_raw)]
        averaged_raw = [round(num, 6) for num in averaged_raw]
        
        metrics = ['MRR', 'MR', 'hit@1', 'hit@3', 'hit@10']
        self.print_result(metrics,"metrics")
        self.print_result(l_raw,"l(raw)")
        self.print_result(r_raw,"r(raw)")
        self.print_result(averaged_raw,"averaged(raw)")

        # 输出指标，往表格里复制的格式
        # MRR hit@1 hit@3 hit@10 
        measure_ori_list = []
        measure_list = []
        for idx,entity in enumerate(zip(l_raw,r_raw,averaged_raw)):
            if idx==1:
                continue
            # print(list(entity))
            
            measure_ori_list.append("  ".join(map(str,list(entity))))
            new_list = [f"{float(item) * 100:.1f}" for item in entity]
            measure_list.append("  ".join(map(str,new_list)))
        print("|".join(measure_ori_list))
        # 在输出一个限制小数点的
        print("|".join(measure_list))
        
        # 返回left_hit10的指标
        hit_n_num = len(list(filter(lambda x: x <= 10, head_rank)))
        return hit_n_num/len(head_rank)
    
    def calc_time_predict(self,time_rank):
        
       
        time_raw = []
        
        
        time_raw.append(self.get_mean_reciprocal_rank(time_rank))
        time_raw.append(self.get_mean_rank(time_rank))
        time_raw.append(self.hit_n(time_rank,1))
        time_raw.append(self.hit_n(time_rank,3))
        time_raw.append(self.hit_n(time_rank,10))
        
       
        metrics = ['MRR', 'MR', 'hit@1', 'hit@3', 'hit@10']
        self.print_result(metrics,"metrics")
        self.print_result(time_raw,"time_raw")
        

        # 输出指标，往表格里复制的格式
        # MRR hit@1 hit@3 hit@10 
        measure_ori_list = []
        measure_list = []
        for idx,entity in enumerate(time_raw):
            if idx==1:
                continue
            # print(list(entity))
            
            measure_ori_list.append(entity)
           
            measure_list.append(f"{float(entity) * 100:.1f}")
        print("|".join(measure_ori_list))
        # 在输出一个限制小数点的
        print("|".join(measure_list))
        
        # 返回left_hit10的指标
        hit_n_num = len(list(filter(lambda x: x <= 10, time_rank)))
        return hit_n_num/len(time_rank)
    
    def run_link_prediction(self,time_constraint_file=None):
       
        head_rank = []
        tail_rank = []
        tp_rank = []
        training_range = tqdm(self.data_loader)
        start = time.time()
        
        
        for index, collate_data in enumerate(self.data_loader):
            data_head = collate_data["data_head"] if "data_head" in collate_data else None
            data_tail = collate_data["data_tail"] if "data_tail" in collate_data else None
            data_tp = collate_data["data_tp"] if "data_tp" in collate_data else None
            if data_head and data_tail:
                
                head_score = self.test_one_step(data_head)
                sorted_head_score = np.argsort(head_score)
                head_position = np.where(sorted_head_score == 0)[0][0] + 1
                head_rank.append(head_position)
                 
                tail_score = self.test_one_step(data_tail)
                sorted_tail_score = np.argsort(tail_score)
                tail_position = np.where(sorted_tail_score == 0)[0][0] + 1
                tail_rank.append(tail_position)
                
            if data_tp:
                print("……")
                
                
            # if (index+1) % 200 == 0:
            #     end = time.time()
            #     print("step: ", index, " ,l:hit_top10_rate: ", self.hit_n(tail_rank,10), " ,mean_rank ", self.get_mean_rank(tail_rank),
            #           'time of testing one triple: %s' % (round((end - start), 3)))
        if data_head and data_tail:
            l_hit10 = self.calc(head_rank,tail_rank)
        if data_tp:
            l_hit10 = self.calc_time_predict(tp_rank)
        return l_hit10

    

    