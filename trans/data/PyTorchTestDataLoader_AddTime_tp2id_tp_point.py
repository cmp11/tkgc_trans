#!/usr/bin/python3
"""
加载测试数据时，需要加载事件发生的起始时间对应的id

返回的数据  batch_h 中是负样本，这个时间跨越了几个时间，batch_h中就有几组负头样本 shape为： [3，time_point_num]
batch_tp shape 为 [3,1] 

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

    def __init__(self, head, tail, rel, st,et,  all_head, all_tail, all_rel, all_tp_point, ent_tot,rel_tot,tp_tot,
               sampling_mode = 'normal',predict_task='entity', bern_flag = False, filter_flag = True):
        # 测试集 triples
        self.head = head
        self.tail = tail
        self.rel = rel
        self.st = st
        self.et = et
  
        self.tri_total = len(head)
  
        # 训练集
        self.all_head = all_head
        self.all_tail = all_tail
        self.all_rel = all_rel
        self.all_tp_point = all_tp_point
  
        # the number of entity and relation
        self.ent_total = ent_tot
        self.rel_total = rel_tot
        self.tp_tot = tp_tot
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
        return (self.head[idx], self.tail[idx], self.rel[idx], self.st[idx],self.et[idx])

    def collate_fn(self, data):
        # 加入时间点
        # 在该函数中确认如何生成负样本

        # https://blog.csdn.net/dong_liuqi/article/details/114521240
        # """collate_fn可以在调用__getitem__函数后，将得到的batch_size个数据进行进一步的处理，在迭代dataloader时，
        # 取出的数据批就是经过了collate_fn函数处理的数据。
          # 换句话说，collate_fn的输入参数是__getitem__的返回值，
        # dataloader的输出是collate_fn的返回值"""
        # 加入时间字段
        data_head = {}
        data_tail = {}
        data_tp = {}
        collate_data = {}
        if self.sampling_mode == "normal":
            batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
            batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
            batch_r = np.array([item[2] for item in data]).reshape(-1, 1)
            if self.predict_task =='entity':
                data_head['mode'] = "normal"
                data_tail['mode'] = "normal"
    
                			
                # batch_st = np.array([item[3] for item in data]).reshape(-1, 1)
                # batch_et = np.array([item[4] for item in data]).reshape(-1, 1)
                
                for index, item in enumerate(data):
                    neg_h_tp_tup_list, neg_t_tp_tup_list,h_batch_tp,t_batch_tp = self.__normal_TransE_batch(item[0], item[1], item[2], item[3], item[4])
                    # neg_h_tp_tup_list 和 h_batch_tp是对应的 负样本对应的时间  均为二维数组
                    neg_h_tp_tup_list, neg_t_tp_tup_list,h_batch_tp,t_batch_tp = self.__normal_TransE_batch(item[0], item[1], item[2], item[3], item[4])
                    if len(neg_h_tp_tup_list) > 0:
                        for i in range(len(neg_h_tp_tup_list)):
                            neg_h_tp_tup_list[i] = np.insert(neg_h_tp_tup_list[i],0,item[0])
                        
                    if len(neg_t_tp_tup_list) > 0:
                        for i in range(len(neg_t_tp_tup_list)):
                            neg_t_tp_tup_list[i] = np.insert(neg_t_tp_tup_list[i], 0, item[1])


                data_head['batch_h'] = neg_h_tp_tup_list
                data_head['batch_t'] = batch_t.squeeze().flatten()
                data_head['batch_r'] = batch_r.squeeze().flatten()
                data_head['batch_tp'] = h_batch_tp
        
                data_tail['batch_h'] = batch_h.squeeze().flatten()
                data_tail['batch_t'] = neg_t_tp_tup_list
                data_tail['batch_r'] = batch_r.squeeze().flatten()
                data_tail['batch_tp'] = t_batch_tp
                
                collate_data["data_head"] = data_head
                collate_data["data_tail"] = data_tail
            else:
                data_tp['mode'] = "normal"
                tmp = np.arange(self.get_tp_tot())
                
            
                data_tp['batch_h'] = batch_h.squeeze().flatten()
                data_tp['batch_t'] = batch_t.squeeze().flatten()
                data_tp['batch_r'] = batch_r.squeeze().flatten()
                data_tp['batch_tp'] = tmp.squeeze().flatten()
                
            
                collate_data["data_tp"] = data_tp
                
        collate_data["test_quintuple"] =  data[0]  
        return collate_data

    

    def __count_htr(self):
        """统计训练集和测试集的head、tail、relation数据
        """
        self.h_of_trtp = {}
        self.t_of_hrtp = {}
        self.r_of_http = {}
        self.t_of_hrt = {}
    

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
            

        for t, r, tp in self.h_of_trtp:
            self.h_of_trtp[(t, r, tp)] = np.array(list(set(self.h_of_trtp[(t, r, tp)])))
        for h, r, tp in self.t_of_hrtp:
            self.t_of_hrtp[(h, r, tp)] = np.array(list(set(self.t_of_hrtp[(h, r, tp)])))
        for h, t, tp in self.r_of_http:
            self.r_of_http[(h, t, tp)] = np.array(list(set(self.r_of_http[(h, t, tp)])))
        

    

    def __corrupt_head_norepeat(self, t, r, st,et):
        
        tmp = np.arange(self.ent_total)
    
        
        # for entity in tmp:
        #     for tp in range(st,et+1):
        #         if entity  not in self.h_of_trtp[(t, r, tp)]:
        #             if entity not in tp_of_h:
        #                 tp_of_h[entity] = []
        #             tp_of_h[entity].append(tp)
        neg_of_h_list = []
        tp_list = []
      
        for tp in range(st,et+1):
            if not self.filter_flag:
                neg = tmp
            else:
                mask = np.in1d(tmp, self.h_of_trtp[(t, r, tp)], assume_unique=True, invert=True)
                neg = tmp[mask]
            neg_of_h_list.append(neg)
            tp_list.append(np.array(tp))
                
        
        return neg_of_h_list,tp_list

    def __corrupt_tail_norepeat(self, h, r,  st,et):
        tmp = np.arange(self.ent_total)
    
        # for entity in tmp:
        #     for tp in range(st,et+1):
        #         if entity  not in self.t_of_hrtp[(t, r, tp)]:
        #             if entity not in tp_of_h:
        #                 tp_of_h[entity] = []
        #             tp_of_h[entity].append(tp)
           
        neg_of_t_list = []
        tp_list = []
        for tp in range(st,et+1):
            if not self.filter_flag:
                neg = tmp
            else:
                mask = np.in1d(tmp, self.t_of_hrtp[(h, r, tp)], assume_unique=True, invert=True)
                neg = tmp[mask]
            neg_of_t_list.append(neg)
            tp_list.append(np.array(tp))
        
                    
       
        return neg_of_t_list,tp_list
        



    def __normal_TransE_batch(self, h, t, r, st,et):
        """
           取负样本的时候根据TransE的策略，取全量的数据
        """
        
    
  
        neg_h_tp_tup_list,h_tp_list = self.__corrupt_head_norepeat(t, r, st,et)
        neg_t_tp_tup_list,t_tp_list = self.__corrupt_tail_norepeat(h, r, st,et)
       
        return neg_h_tp_tup_list, neg_t_tp_tup_list,h_tp_list,t_tp_list

    

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
        predict_task = "entity",
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
            
  
        
        dataset = self.__construct_dataset(sampling_mode, predict_task, bern_flag, filter_flag)

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
                

    def __construct_dataset(self, sampling_mode, predict_task,bern_flag, filter_flag):
        
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
        st_list = []
        et_list = []
          # 加载测试数据
        f = open(self.test_tri_file, "r")
        test_triples_total = (int)(f.readline())
        for index in range(test_triples_total):
            # 注意需要与数据集文件每列的含义对齐
            h,r,t, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
            # h, r, t, tp = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
            head_tp.append((int)(h))
            tail_tp.append((int)(t))
            rel_tp.append((int)(r))
            st_list.append(int(st))
            et_list.append(int(et))

        f.close()
        
        # 加载所有数据的三元组	
        f = open(self.tri_file, "r")
        triples_total = (int)(f.readline())
        for index in range(triples_total):
            text = f.readline()
            h, r, t, tp= text.strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
                
            all_head.append((int)(h))
            all_tail.append((int)(t))
            all_rel.append((int)(r))
            all_tp_point.append((int)(tp))
            


        dataset = PyTorchTestDataset(np.array(head_tp), np.array(tail_tp), np.array(rel_tp), np.array(st_list),np.array(et_list),
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
    def get_data_path(self):
        return self.in_path
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
