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
'''
测试方法，将事实分割为时间点，
实体链接预测 计算指标时，将每个实体对应时间点得分加权平均值进行排名
时间预测   计算指标时，按照论文上写的accard Coefficient系数平均值
'''

class Tester_AddTime(object):

    def __init__(self, model = None, data_loader = None, use_gpu = True,constraint_file = None):
        """_summary_

        Args:
            model (_type_, optional): _description_. Defaults to None.
            data_loader (_type_, optional): _description_. Defaults to None.
            use_gpu (bool, optional): _description_. Defaults to True.
            constraint_file (_type_, optional): _description_. Defaults to None. 只有在时间预测的时候需要指定参数，指定约束文件
        """
        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.rel_time_constrain = {}
        if self.use_gpu:
            self.model.cuda()
            
        for i in range(self.data_loader.get_rel_tot()):
            self.rel_time_constrain[i] = 1
        if constraint_file:
            self.read_time_constrain(constraint_file)
       
    def read_time_constrain(self , constraint_file):
        with open(constraint_file,"r") as file:
            totle = int(file.readline())
            for i in range(totle):
                r,duration = file.readline().strip().split()
                self.rel_time_constrain[int(r)] = int(duration)
        
    
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
    
      
    def run_link_prediction(self):
       
        head_rank = []
        tail_rank = []
        Jaccard_Coefficient_J_list = []
        Jaccard_Coefficient_K_list = []
        
        training_range = tqdm(self.data_loader)
        start = time.time()
        
        
        for index, collate_data in enumerate(self.data_loader):
            data_head = collate_data["data_head"] if "data_head" in collate_data else None
            data_tail = collate_data["data_tail"] if "data_tail" in collate_data else None
            data_tp = collate_data["data_tp"] if "data_tp" in collate_data else None
            test_quintuple = collate_data["test_quintuple"]
           
            
            event_h,event_t,event_r, event_st ,event_et = test_quintuple
            if data_head and data_tail:
               
                head_event_score_dict = {}
                tail_event_score_dict = {}

                head_batch_h = data_head["batch_h"]
                head_batch_tp = data_head["batch_tp"]
                
                tail_batch_t = data_tail["batch_t"]
                tail_batch_tp = data_tail["batch_tp"] 
                
                
                
                for batch_h,tp in zip(head_batch_h,head_batch_tp):
                    # print("----------------")
                    # print("tp : " , tp)
                    data_head["batch_h"] = batch_h.squeeze().flatten()
                    data_head["batch_tp"] = tp.squeeze().flatten()
                    head_score = self.test_one_step(data_head) 
                    head_filter = set()
                    tail_filter = set()
                    for h,score in zip(batch_h,head_score):
                        if (h,event_r,event_t) in head_filter:
                            continue
                        head_filter.add((h,event_r,event_t))
                        
                        if h not in head_event_score_dict:
                            head_event_score_dict[h] = []
                        head_event_score_dict[h].append(round(score,4))
                
                    
                for batch_t,tp in zip(tail_batch_t,tail_batch_tp):
                    data_tail["batch_t"] = batch_t.squeeze().flatten()
                    data_tail["batch_tp"] = tp.squeeze().flatten()
                    tail_score = self.test_one_step(data_tail) 
                    head_filter = set()
                    tail_filter = set()
                    for t,score in zip(batch_t,tail_score):
                        if (event_h,event_r,t) in tail_filter:
                            continue
                        tail_filter.add((event_h,event_r,t))
                        
                        if t not in tail_event_score_dict:
                            tail_event_score_dict[t] = []
                        
                        tail_event_score_dict[t].append(round(score,4))            

                # 对每一个事件的负样本 计算平均值
                # head_event_score_dict
                head_event_score_dict = {key: round(np.mean(value),4) for key, value in head_event_score_dict.items()}
                tail_event_score_dict = {key: round(np.mean(value),4) for key, value in tail_event_score_dict.items()}
                # 获取正确样本对应的分值
                # print("test_triple: ",test_triple)
                
                
                h_test_score = head_event_score_dict[event_h]
                t_test_score = tail_event_score_dict[event_t]

                head_event_score_sorted = sorted(head_event_score_dict.values())
                tail_event_score_sorted = sorted(tail_event_score_dict.values())
                
                # 计算排名
                head_ranking = head_event_score_sorted.index(h_test_score) + 1
                tail_ranking = tail_event_score_sorted.index(t_test_score) + 1
                
                head_rank.append(head_ranking)
                tail_rank.append(tail_ranking)

                # if (index+1) % 200 == 0:
                #     end = time.time()
                #     print("step: ", index, " ,l:hit_top10_rate: ", self.hit_n(tail_rank,10), " ,mean_rank ", self.get_mean_rank(tail_rank),
                #           'time of testing one triple: %s' % (round((end - start), 3)))
                head_filter.clear
                tail_filter.clear
                
            if data_tp:
                tp_score = self.test_one_step(data_tp)
                sorted_tp_score = np.argsort(tp_score)
                tp_mean_num_of_r = self.rel_time_constrain[event_r]
                hit_quintuple_tp_index = sorted_tp_score[:tp_mean_num_of_r]
                
                # 预测的事件id集合
                t_pr = [data_tp['batch_tp'][i] for i in hit_quintuple_tp_index]
                # 真实的事件id集合
                t_re = list(range(event_st,event_et+1))
                
                # 计算两个集合的交集
                intersection = set(t_pr) & set(t_re)

                # 计算两个集合的并集
                union = set(t_pr) | set(t_re)
                
                Jaccard_Coefficient_J = round(len(intersection)/len(union),4)
                Jaccard_Coefficient_K = round(len(intersection)/(len(t_pr)*len(t_re)) ** 0.5,4)
                Jaccard_Coefficient_J_list.append(Jaccard_Coefficient_J)
                Jaccard_Coefficient_K_list.append(Jaccard_Coefficient_K)
                # print("Jaccard_Coefficient_J: ",Jaccard_Coefficient_J)
                # print("Jaccard_Coefficient_K: ",Jaccard_Coefficient_K)

        if data_head and data_tail:
            l_hit10 = self.calc(head_rank,tail_rank)
            return l_hit10
        if data_tp:
            Jaccard_Coefficient_J_mean = round(sum(Jaccard_Coefficient_J_list)/len(Jaccard_Coefficient_J_list),4)
            Jaccard_Coefficient_K_mean = round(sum(Jaccard_Coefficient_K_list)/len(Jaccard_Coefficient_K_list),4)
            print("Jaccard_Coefficient_J_mean : ",Jaccard_Coefficient_J_mean)
            print("Jaccard_Coefficient_K_mean :",Jaccard_Coefficient_K_mean)
            return sum(Jaccard_Coefficient_J_list)/len(Jaccard_Coefficient_J_list)

    

    