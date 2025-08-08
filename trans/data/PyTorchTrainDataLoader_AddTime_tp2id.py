#!/usr/bin/python3
"""
加载训练集  数据集中已经将事件时间拆解为时间点 有了tp2id.txt文件
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PyTorchTrainDataset(Dataset):

    def __init__(self, head, tail, rel, tp_point, 
              ent_total, rel_total,tp_total, sampling_mode = 'normal',sampling_choice=None, bern_flag = False, filter_flag = True, neg_ent = 1, neg_rel = 0,
              neg_time=0):
        # triples
        self.head = head
        self.tail = tail
        self.rel = rel
        self.tp_point = tp_point
        # total numbers of entities, relations, and triples
        self.rel_total = rel_total
        self.ent_total = ent_total
        self.tp_total = tp_total
  
        self.tri_total = len(head)
        
        # the sampling mode
        self.sampling_mode = sampling_mode # 数据s采样模式，normal 表示正常负采样，cross 表示交替替换 head 和 tail 进行负采样
        
        self.sampling_choice = sampling_choice  # 决定使用transe的负取样方法还是使用transH的负取样方法(计算概率)
        # the number of negative examples
        self.neg_ent = neg_ent
        self.neg_rel = neg_rel
        self.neg_time = neg_time
        self.bern_flag = bern_flag # 是否使用 TransH 提出的负采样方法进行负采样
        self.filter_flag = filter_flag
        if self.sampling_mode == "normal":
            self.cross_sampling_flag = None
        else:
            self.cross_sampling_flag = 0
        try:
            self.__count_htr()
        except Exception as  e:
            print(e)
      
        

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
        batch_data = {}
        if self.sampling_mode == "normal":
            batch_data['mode'] = "normal"
            batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
            batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
            batch_r = np.array([item[2] for item in data]).reshape(-1, 1)			
            batch_tp = np.array([item[3] for item in data]).reshape(-1, 1)
    
            batch_h = np.repeat(batch_h, 1 + self.neg_ent + self.neg_rel + self.neg_time, axis = -1)
            batch_t = np.repeat(batch_t, 1 + self.neg_ent + self.neg_rel+ self.neg_time, axis = -1)
            batch_r = np.repeat(batch_r, 1 + self.neg_ent + self.neg_rel+ self.neg_time, axis = -1)			
            batch_tp = np.repeat(batch_tp, 1 + self.neg_ent + self.neg_rel+ self.neg_time, axis = -1)
    
            for index, item in enumerate(data):
                last = 1
                if self.neg_ent > 0:
                    neg_head, neg_tail = self.__normal_batch(item[0], item[1], item[2], self.neg_ent, self.sampling_choice)
                    if len(neg_head) > 0:
                        batch_h[index][last:last + len(neg_head)] = neg_head
                        last += len(neg_head)
                    if len(neg_tail) > 0:
                        batch_t[index][last:last + len(neg_tail)] = neg_tail
                        last += len(neg_tail)
                if self.neg_rel > 0:
                    
                    neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
                    batch_r[index][last:last + len(neg_rel)] = neg_rel
                # todo 是否生成时间字段的负样本

                if self.neg_time > 0:
                    # neg_time = self.__time_batch(3483, 3484, 14, self.neg_time)
                    neg_time = self.__time_batch(item[0], item[1], item[2], self.neg_time)
                    batch_tp[index][last:last + len(neg_time)] = neg_time
            batch_h = batch_h.transpose()
            batch_t = batch_t.transpose()
            batch_r = batch_r.transpose()
            batch_tp = batch_tp.transpose()
                
            
        else:
              # 交替替换 head 和 tail 进行负采样, 生成 1 batch 数据
            self.cross_sampling_flag = 1 - self.cross_sampling_flag
            if self.cross_sampling_flag == 0:
                batch_data['mode'] = "head_batch"
                batch_h = np.array([[item[0]] for item in data])
                batch_t = np.array([item[1] for item in data])
                batch_r = np.array([item[2] for item in data])
                batch_h = np.repeat(batch_h, 1 + self.neg_ent, axis = -1)
                for index, item in enumerate(data):
                    neg_head = self.__head_batch(item[0], item[1], item[2], self.neg_ent)
                    batch_h[index][1:] = neg_head
                batch_h = batch_h.transpose()
            else:
                batch_data['mode'] = "tail_batch"
                batch_h = np.array([item[0] for item in data]) 
                batch_t = np.array([[item[1]] for item in data])
                batch_r = np.array([item[2] for item in data])
                batch_t = np.repeat(batch_t, 1 + self.neg_ent, axis = -1)
                for index, item in enumerate(data):
                    neg_tail = self.__tail_batch(item[0], item[1], item[2], self.neg_ent)
                    batch_t[index][1:] = neg_tail
                batch_t = batch_t.transpose()

        batch_y = np.concatenate([np.ones((len(data), 1)), np.zeros((len(data), self.neg_ent + self.neg_rel + self.neg_time))], -1).transpose()
        batch_data['batch_h'] = batch_h.squeeze().flatten()
        batch_data['batch_t'] = batch_t.squeeze().flatten()
        batch_data['batch_r'] = batch_r.squeeze().flatten()
        batch_data['batch_tp'] = batch_tp.squeeze().flatten()
        batch_data['batch_y'] = batch_y.squeeze().flatten()
        
        return batch_data

    def __count_htr(self):
        """统计head、tail、relation数据
        """
        self.h_of_tr = {}
        self.t_of_hr = {}
        self.r_of_ht = {}
        self.h_of_r = {}
        self.t_of_r = {}
        self.freqRel = {}
        self.lef_mean = {}
        self.rig_mean = {}
  
        self.tp_of_htr = {}

        triples = zip(self.head, self.tail, self.rel,self.tp_point)
        for h, t, r, tp in triples:
            if (h, r) not in self.t_of_hr:
                self.t_of_hr[(h, r)] = []
            self.t_of_hr[(h, r)].append(t)
            if (t, r) not in self.h_of_tr:
                self.h_of_tr[(t, r)] = []
            self.h_of_tr[(t, r)].append(h)
            if (h, t) not in self.r_of_ht:
                self.r_of_ht[(h, t)] = []
            self.r_of_ht[(h, t)].append(r)
   
            if (h, t, r) not in self.tp_of_htr:
                self.tp_of_htr[(h, t, r)] = []
            self.tp_of_htr[(h, t, r)].append(tp)
   
            if r not in self.freqRel:
                self.freqRel[r] = 0
                self.h_of_r[r] = {}
                self.t_of_r[r] = {}
            self.freqRel[r] += 1.0
            self.h_of_r[r][h] = 1
            self.t_of_r[r][t] = 1

        for t, r in self.h_of_tr:
            self.h_of_tr[(t, r)] = np.array(list(set(self.h_of_tr[(t, r)])))
        for h, r in self.t_of_hr:
            self.t_of_hr[(h, r)] = np.array(list(set(self.t_of_hr[(h, r)])))
        for h, t in self.r_of_ht:
            self.r_of_ht[(h, t)] = np.array(list(set(self.r_of_ht[(h, t)])))
   
        for h, t, r in self.tp_of_htr:
            self.tp_of_htr[(h, t, r)] = np.array(list(set(self.tp_of_htr[(h, t, r)])))
   
        for r in range(self.rel_total):
            self.h_of_r[r] = np.array(list(self.h_of_r[r].keys()))
            self.t_of_r[r] = np.array(list(self.t_of_r[r].keys()))
            self.lef_mean[r] = self.freqRel[r] / len(self.h_of_r[r])
            self.rig_mean[r] = self.freqRel[r] / len(self.t_of_r[r])

    def __corrupt_head(self, t, r, num_max = 1):
        tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.h_of_tr[(t, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_tail(self, h, r, num_max = 1):
        tmp = torch.randint(low = 0, high = self.ent_total, size = (num_max, )).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.t_of_hr[(h, r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg


    def __corrupt_rel(self, h, t, num_max = 1):
        tmp = torch.randint(low = 0, high = self.rel_total, size = (num_max, )).numpy()
        if not self.filter_flag:
            return tmp
        mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg

    def __corrupt_time(self, h,t,  r,num_max = 1):
        # tmp = torch.randint(low = 0, high = self.tp_total, size = (num_max, )).numpy()
        tmp = torch.randperm(self.tp_total).numpy()
        if not self.filter_flag:
            return tmp
       
        mask = np.in1d(tmp, self.tp_of_htr[(h, t,r)], assume_unique=True, invert=True)
        neg = tmp[mask]
        return neg
    
    def __normal_batch(self, h, t, r, neg_size, method=None):
        neg_size_h = 0
        neg_size_t = 0
        
        if not method:
              # transH的负采样策略
            prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
            for i in range(neg_size):
                if random.random() < prob:
                    neg_size_h += 1
                else:
                    neg_size_t += 1
        else:
            neg_size_h = neg_size_t = int(neg_size/2)
            

        neg_list_h = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_h:
            neg_tmp_h = self.__corrupt_head(t, r, num_max = (neg_size_h - neg_cur_size) * 2)
            neg_list_h.append(neg_tmp_h)
            neg_cur_size += len(neg_tmp_h)
        if neg_list_h != []:
            neg_list_h = np.concatenate(neg_list_h)
            
        neg_list_t = []
        neg_cur_size = 0
        while neg_cur_size < neg_size_t:
            neg_tmp_t = self.__corrupt_tail(h, r, num_max = (neg_size_t - neg_cur_size) * 2)
            neg_list_t.append(neg_tmp_t)
            neg_cur_size += len(neg_tmp_t)
        if neg_list_t != []:
            neg_list_t = np.concatenate(neg_list_t)

        return neg_list_h[:neg_size_h], neg_list_t[:neg_size_t]

    def __head_batch(self, h, t, r, neg_size):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_head(t, r, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __tail_batch(self, h, t, r, neg_size):
        # return torch.randint(low = 0, high = self.ent_total, size = (neg_size, )).numpy()
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_tail(h, r, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __rel_batch(self, h, t, r, neg_size):
        neg_list = []
        neg_cur_size = 0
        while neg_cur_size < neg_size:
            neg_tmp = self.__corrupt_rel(h, t, num_max = (neg_size - neg_cur_size) * 2)
            neg_list.append(neg_tmp)
            neg_cur_size += len(neg_tmp)
        return np.concatenate(neg_list)[:neg_size]

    def __time_batch(self, h, t, r, neg_size):
       
        neg_tmp = self.__corrupt_time(h, t,r, num_max = self.tp_total)
        if len(neg_tmp) >= neg_size:
            selected = np.random.choice(neg_tmp, size=10, replace=False)
            return selected
        if len(neg_tmp)==0 :
            print(h,r,t) 
            raise ValueError("The number of negative samples is 0")

        else:
            selected = list(neg_tmp)
            remaining_needed = 10 - len(neg_tmp)
            # 从 tmp 中随机选择剩余需要的值，允许重复
            additional_values = np.random.choice(neg_tmp, size=remaining_needed, replace=True)
            selected.extend(additional_values)
            return selected
        

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

    def get_tp_tot(self):
        return self.tp_total

    def get_tri_tot(self):
        return self.tri_total

    def get_tp_max(self):
        return max(self.tp_point)

    def get_tp_min(self):
        return min(self.tp_point)


    

    

class PyTorchTrainDataLoader(DataLoader):

    """
        数据加载时，时间处理成时间点，按照实体和关系的形式处理为tp2id的形式。
     """
    def __init__(self, 
        in_path = None, 
        train_file = None,
        ent_file = None,
        rel_file = None,
        tp_file = None,
        batch_size = None, 
        nbatches = None, 
        threads = 8, 
        sampling_mode = "normal",
        
        sampling_choice = None, # 在选择取样方式时，None 表示就是最普通的，正样本和负样本各取一半，不为None时，说明使用TransH的方式，用概率来进行取样
        bern_flag = False, # bern_flag跟sampling_choice很相似，只不过bern_flag是概率0.5，sampling_choice保证一定是0.5
        filter_flag = True, # 是否对负取样的数据进行过滤，True保证负样本不在训练集中出现，False表示不做这个过滤
        neg_ent = 1, 
        neg_rel = 0,
          neg_time =0,
        # time_point_formate = "%Y",  # %Y year;  %Y-%m month;  %Y-%m day
        shuffle = True, 
        drop_last = True):

        # self.st_dft = "1900"
        # self.et_dft = "2024"

        self.in_path = in_path
        self.train_file = train_file
        self.ent_file = ent_file
        self.rel_file = rel_file
        self.tp_file = tp_file
        if in_path != None:
            self.train_file = in_path + "train2id.txt"
            self.ent_file = in_path + "entity2id.txt"
            self.rel_file = in_path + "relation2id.txt"
            self.tp_file = in_path + "tp2id.txt"

        dataset = self.__construct_dataset(sampling_mode,sampling_choice, bern_flag, filter_flag, neg_ent, neg_rel,neg_time)

        self.batch_size = batch_size
        self.nbatches = nbatches
        if batch_size == None:
            self.batch_size = dataset.get_tri_tot() // nbatches
        if nbatches == None:
            self.nbatches = dataset.get_tri_tot() // batch_size

        super(PyTorchTrainDataLoader, self).__init__(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = threads,
            collate_fn = dataset.collate_fn,
            drop_last = drop_last)
    
            

    def __construct_dataset(self, sampling_mode,sampling_choice, bern_flag, filter_flag, neg_ent, neg_rel,neg_time):
        f = open(self.ent_file, "r")
        ent_total = (int)(f.readline())
        f.close()

        f = open(self.rel_file, "r")
        rel_total = (int)(f.readline())
        f.close()
  
        f = open(self.tp_file, "r")
        tp_total = (int)(f.readline())
        f.close()

        head_tp = []
        tail_tp = []
        rel_tp = []
        tp_point = []

        triple_set=[]

        f = open(self.train_file, "r")
        train_total = (int)(f.readline())
        for index in range(train_total):
            # 注意需要与数据集文件每列的含义对齐
            # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
            text = f.readline()
            
            h, r, t, tp,*et= text.strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
            
            
            head_tp.append((int)(h))
            tail_tp.append((int)(t))
            rel_tp.append((int)(r))
            tp_point.append(int(tp))

        f.close()


            
            
        # triple_set = set(triple_set)
        dataset = PyTorchTrainDataset(np.array(head_tp), np.array(tail_tp), np.array(rel_tp), np.array(tp_point), 
                                ent_total, rel_total, tp_total,sampling_mode,sampling_choice, bern_flag, filter_flag, neg_ent, neg_rel,neg_time)
  
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
    date_list = PyTorchTrainDataLoader.getEveryDay('2016-01','2019-03', date_formate)
    # print(sorted(set(date_list)))
    print(date_list)
