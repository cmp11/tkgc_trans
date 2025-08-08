#!/usr/bin/python3
"""
加载时间点
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PyTorchTestDataset(Dataset):

    def __init__(self, head, tail, rel, tp_point,  all_head, all_tail, all_rel, all_tp_point, ent_tot,rel_tot,tp_tot,
               sampling_mode = 'normal', predict_task = "entity",bern_flag = False, filter_flag = True):
        # 测试集 triples
        self.head = head
        self.tail = tail
        self.rel = rel
        self.tp_point = tp_point
        
        self.tri_total = len(head)
  
        # 训练集
        self.all_head = all_head
        self.all_tail = all_tail
        self.all_rel = all_rel
        self.all_tp_point = all_tp_point
  
        # the number of entity and relation
        self.ent_total = ent_tot
        self.rel_total = rel_tot
        self.tp_total = tp_tot
        # the sampling mode
        self.sampling_mode = sampling_mode # 数据s采样模式，normal 表示正常负采样，cross 表示交替替换 head 和 tail 进行负采样
        self.predict_task = predict_task
        # the number of negative examples
        # self.neg_ent = neg_ent
        # self.neg_rel = neg_rel 
        # self.neg_time = neg_time
        self.bern_flag = bern_flag # 是否使用 TransH 提出的负采样方法进行负采样(使用概率生成正负样本)
        self.filter_flag = filter_flag  # 是否筛选生成的负样本是假阳（也作为正样本存在）
        if self.sampling_mode == "normal":
            self.cross_sampling_flag = None
        else:
            self.cross_sampling_flag = 0
        self.__count_htr()

    def __len__(self):
        return self.tri_total

    def __getitem__(self, idx):
        # 加入时间字段
        return (self.head[idx], self.tail[idx], self.rel[idx], self.tp_point[idx])

    def collate_fn(self, data):
        # 加入时间点
        # 在该函数中确认如何生成负样本

        # https://blog.csdn.net/dong_liuqi/article/details/114521240
        # """collate_fn可以在调用__getitem__函数后，将得到的batch_size个数据进行进一步的处理，在迭代dataloader时，
        # 取出的数据批就是经过了collate_fn函数处理的数据。
          # 换句话说，collate_fn的输入参数是__getitem__的返回值，
        # dataloader的输出是collate_fn的返回值"""
        # 加入时间字段
        collate_data = {}
        data_head = {}
        data_tail = {}
        data_rel = {}
        data_tp = {}
        if self.sampling_mode == "normal":
            if self.predict_task =='entity':
                data_head['mode'] = "normal"
                data_tail['mode'] = "normal"
    
                batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
                batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
                batch_r = np.array([item[2] for item in data]).reshape(-1, 1)			
                batch_tp = np.array([item[3] for item in data]).reshape(-1, 1)
            
                
                for index, item in enumerate(data):
                    
                    neg_head, neg_tail = self.__normal_TransE_batch(item[0], item[1], item[2], item[3])
                    if len(neg_head) > 0:
                        batch_neg_head = neg_head
                        batch_neg_head = np.insert(batch_neg_head, 0, item[0])
                            
                    if len(neg_tail) > 0:
                        batch_neg_tail = neg_tail
                        batch_neg_tail = np.insert(batch_neg_tail, 0, item[1])
               
        
                data_head['batch_h'] = batch_neg_head.squeeze().flatten()
                data_head['batch_t'] = batch_t.squeeze().flatten()
                data_head['batch_r'] = batch_r.squeeze().flatten()
                data_head['batch_tp'] = batch_tp.squeeze().flatten()
        
                data_tail['batch_h'] = batch_h.squeeze().flatten()
                data_tail['batch_t'] = batch_neg_tail.squeeze().flatten()
                data_tail['batch_r'] = batch_r.squeeze().flatten()
                data_tail['batch_tp'] = batch_tp.squeeze().flatten()
                
                collate_data["data_head"] = data_head
                collate_data["data_tail"] = data_tail

            else:
                print("...")
               
    
        collate_data["test_quintuple"] =  data[0]          
        return collate_data

    

    def __count_htr(self):
        """统计训练集和测试集的head、tail、relation数据
        """
        self.h_of_trtp = {}
        self.t_of_hrtp = {}
        self.r_of_http = {}
        self.tp_of_htr = {}
    

        triples = zip(self.all_head, self.all_tail, self.all_rel,self.all_tp_point)
        for h, t, r,tp in triples:
            if (h, r, tp) not in self.t_of_hrtp:
                self.t_of_hrtp[(h, r, tp)] = []
            self.t_of_hrtp[(h, r, tp)].append(t)
            if (t, r, tp) not in self.h_of_trtp:
                self.h_of_trtp[(t, r, tp)] = []
            self.h_of_trtp[(t, r, tp)].append(h)
            if (h, t, tp) not in self.r_of_http:
                self.r_of_http[(h, t, tp)] = []
            self.r_of_http[(h, t, tp)].append(r)
            
            if (h, t, r) not in self.tp_of_htr:
                self.tp_of_htr[(h, t, r)] = []
            self.tp_of_htr[(h, t, r)].append(tp)
            

        for t, r, tp in self.h_of_trtp:
            self.h_of_trtp[(t, r, tp)] = np.array(list(set(self.h_of_trtp[(t, r, tp)])))
        for h, r, tp in self.t_of_hrtp:
            self.t_of_hrtp[(h, r, tp)] = np.array(list(set(self.t_of_hrtp[(h, r, tp)])))
        for h, t, tp in self.r_of_http:
            self.r_of_http[(h, t, tp)] = np.array(list(set(self.r_of_http[(h, t, tp)])))
        

    

    def __corrupt_head_norepeat(self, t, r, tp):
        tmp = np.arange(self.ent_total)
        if not self.filter_flag or  (t, r, tp) not in self.h_of_trtp:
            return tmp
        mask = np.in1d(tmp, self.h_of_trtp[(t, r, tp)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_tail_norepeat(self, h, r, tp):
        tmp = np.arange(self.ent_total)
        if not self.filter_flag or (h, r, tp) not in self.t_of_hrtp:
            return tmp
        mask = np.in1d(tmp, self.t_of_hrtp[(h, r, tp)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_rel_norepeat(self, h, t, tp):
        tmp = np.arange(self.rel_total)
        
        if not self.filter_flag or (h, t, tp) not in self.r_of_http:
            return tmp
        mask = np.in1d(tmp, self.r_of_http[(h, t, tp)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

  
    
    def __corrupt_time_norepeat(self,h, t, r):
        tmp = np.arange(self.tp_total)
        if not self.filter_flag or  (h, t, r) not in self.tp_of_htr:
            return tmp
        mask = np.in1d(tmp, self.tp_of_htr[(h, t, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __normal_TransE_batch(self, h, t, r, tp):
        """
           取负样本的时候根据TransE的策略，取全量的数据
        """
        
        neg_list_h = []
        neg_list_t = []
  
        neg_tmp_h = self.__corrupt_head_norepeat(t, r, tp)
        neg_tmp_h = neg_tmp_h[neg_tmp_h != h]
        neg_list_h.append(neg_tmp_h)
        
        
        neg_tmp_t = self.__corrupt_tail_norepeat(h, r, tp)
        neg_tmp_t = neg_tmp_t[neg_tmp_t != t]
        neg_list_t.append(neg_tmp_t)
        
            
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
            
   
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_h, neg_list_t

    

    def set_sampling_mode(self, sampling_mode):
        self.sampling_mode = sampling_mode

    def set_ent_neg_rate(self, rate):
        self.neg_ent = rate

    def set_rel_neg_rate(self, rate):
        self.neg_rel = rate

    def set_bern_flag(self, bern_flag):
        self.bern_flag = bern_flag

    def set_filter_flag(self, filter_flag):
        self.filter_flag = filter_flag

    def get_ent_tot(self):
        return self.ent_total

    def get_rel_tot(self):
        return self.rel_total

    def get_tri_tot(self):
        return self.tri_total

    def get_tp_max(self):
        return max(self.tp_point)


    def get_tp_tot(self):
        return self.tp_tot
    

    

class PyTorchTestDataLoader(DataLoader):

    def __init__(self, 
        in_path = None, 
        tri_file = None,
        test_tri_file = None,
        threads = 8, 
        sampling_mode = "normal", 
        predict_task="entity",
        bern_flag = False, 
        filter_flag = True, 
        ent_tot=0,
        rel_tot=0,
        tp_tot = 0,
        # neg_ent = 1, 
        # neg_rel = 0, 
        shuffle = True, 
        drop_last = True):



        self.in_path = in_path
        self.tri_file = tri_file
        self.test_tri_file = test_tri_file
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.tp_tot = tp_tot

        
        if in_path != None:
      
            self.test_tri_file = in_path + "test.txt"
            self.tri_file = in_path + "triple2id.txt"
            
  
        
        dataset = self.__construct_dataset(sampling_mode,predict_task, bern_flag, filter_flag)

        batch_size = 1
        self.nbatches = dataset.get_tri_tot() // batch_size
        
            

        super(PyTorchTestDataLoader, self).__init__(
            dataset = dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = threads,
            collate_fn = dataset.collate_fn,
            drop_last = drop_last)
                

    def __construct_dataset(self, sampling_mode,predict_task, bern_flag, filter_flag):
        
        # 合并训练集和测试集中的头结点和尾节点和关系
        all_head = []
        all_tail = []
        all_rel = []
        all_tp_point = [] 
        # start_time = []
        # end_time = []
        head_tp = []
        tail_tp = []
        rel_tp = []
        tp_point = []

          # 加载测试数据
        f = open(self.test_tri_file, "r")
        test_triples_total = (int)(f.readline())
        for index in range(test_triples_total):
            # 注意需要与数据集文件每列的含义对齐
            # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
            h, r, t, tp, *et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
            head_tp.append((int)(h))
            tail_tp.append((int)(t))
            rel_tp.append((int)(r))
            tp_point.append(int(tp))

        # 加载所有数据的三元组	
        f = open(self.tri_file, "r")
        triples_total = (int)(f.readline())
        for index in range(triples_total):
            h, r, t, tp, *et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
                
            all_head.append((int)(h))
            all_tail.append((int)(t))
            all_rel.append((int)(r))
            all_tp_point.append((int)(tp))
            


        dataset = PyTorchTestDataset(np.array(head_tp), np.array(tail_tp), np.array(rel_tp), np.array(tp_point),
                               np.array(all_head), np.array(all_tail), np.array(all_rel), np.array(all_tp_point),
                               self.ent_tot,self.rel_tot,self.tp_tot,
                            sampling_mode, predict_task,bern_flag, filter_flag)
  
        return dataset

    """interfaces to set essential parameters"""

    def set_sampling_mode(self, sampling_mode):
        self.dataset.set_sampling_mode(sampling_mode)

    def set_work_threads(self, work_threads):
        self.num_workers = work_threads

    def set_nbatches(self, nbatches):
        self.nbatches = nbatches
        self.batch_size = self.tripleTotal // self.nbatches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.nbatches = self.tripleTotal // self.batch_size

    def set_ent_neg_rate(self, rate):
        self.dataset.set_ent_neg_rate(rate)

    def set_rel_neg_rate(self, rate):
        self.dataset.set_rel_neg_rate(rate)

    def set_bern_flag(self, bern_flag):
        self.dataset.set_bern_flag(bern_flag)

    def set_filter_flag(self, filter_flag):
        self.dataset.set_filter_flag(filter_flag)
    
    """interfaces to get essential parameters"""
    
    def get_batch_size(self):
        return self.batch_size

    def get_ent_tot(self):
        return self.dataset.get_ent_tot()

    def get_rel_tot(self):
        return self.dataset.get_rel_tot()

    def get_triple_tot(self):
        return self.dataset.get_tri_tot()

    def get_tp_tot(self):
        return self.dataset.get_tp_tot()




if __name__ == '__main__':
    year_formate = "%Y"
    month_formate = "%Y-%m"
    day_formte = "%Y-%m-%d"
    date_formate = month_formate
    date_list = PyTorchTestDataLoader.getEveryDay('2016-01','2019-03', date_formate)
    # print(sorted(set(date_list)))
    print(date_list)
