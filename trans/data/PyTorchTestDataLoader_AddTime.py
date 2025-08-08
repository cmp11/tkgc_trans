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

	def __init__(self, head, tail, rel, tp_point,  all_head, all_tail, all_rel, all_tp_point, ent_tot,rel_tot,
               sampling_mode = 'normal', bern_flag = False, filter_flag = True):
		# 测试集 triples
		self.head = head
		self.tail = tail
		self.rel = rel
		self.start_time = tp_point
		
		self.tri_total = len(head)
  
		# 训练集
		self.all_head = all_head
		self.all_tail = all_tail
		self.all_rel = all_rel
		self.all_tp_point = all_tp_point
  
		# the number of entity and relation
		self.ent_total = ent_tot
		self.rel_total = rel_tot
  
		# the sampling mode
		self.sampling_mode = sampling_mode # 数据s采样模式，normal 表示正常负采样，cross 表示交替替换 head 和 tail 进行负采样
		# the number of negative examples
		# self.neg_ent = neg_ent
		# self.neg_rel = neg_rel 
		# self.neg_time = neg_time
		self.bern_flag = bern_flag # 是否使用 TransH 提出的负采样方法进行负采样
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
		return (self.head[idx], self.tail[idx], self.rel[idx], self.start_time[idx])

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
		data_rel = {}
		data_tp = {}
		if self.sampling_mode == "normal":
			data_head['mode'] = "normal"
			data_tail['mode'] = "normal"
   
			batch_h = np.array([item[0] for item in data]).reshape(-1, 1)
			batch_t = np.array([item[1] for item in data]).reshape(-1, 1)
			batch_r = np.array([item[2] for item in data]).reshape(-1, 1)			
			batch_tp = np.array([item[3] for item in data]).reshape(-1, 1)
   			
			
			for index, item in enumerate(data):
				
				neg_head, neg_tail = self.__normal_TransE_batch(item[0], item[1], item[2])
				if len(neg_head) > 0:
					batch_neg_head = neg_head
					batch_neg_head = np.insert(batch_neg_head, 0, item[0])
						
				if len(neg_tail) > 0:
					batch_neg_tail = neg_tail
					batch_neg_tail = np.insert(batch_neg_tail, 0, item[1])
					
				# if self.neg_rel > 0:
				# 	neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
				# 	batch_neg_r = neg_rel
				# 	batch_neg_r = np.insert(batch_neg_r, 0, item[2])
				# # todo 是否生成时间字段的负样本
				# if self.neg_time > 0:
				# 	pass
		
		
	
		data_head['batch_h'] = batch_neg_head.squeeze().flatten()
		data_head['batch_t'] = batch_t.squeeze().flatten()
		data_head['batch_r'] = batch_r.squeeze().flatten()
		data_head['batch_tp'] = batch_tp.squeeze().flatten()
  
		data_tail['batch_h'] = batch_h.squeeze().flatten()
		data_tail['batch_t'] = batch_neg_tail.squeeze().flatten()
		data_tail['batch_r'] = batch_r.squeeze().flatten()
		data_tail['batch_tp'] = batch_tp.squeeze().flatten()
		
		return [data_head,data_tail]

	def collate_fn_bk(self, data):
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
   			
			# batch_h = np.pad(batch_h, ((0, 0), (0, self.neg_ent)), mode='constant', constant_values=1)
			# batch_t = np.pad(batch_t, ((0, 0), (0, self.neg_ent)), mode='constant', constant_values=1)
			# batch_r = np.pad(batch_r, ((0, 0), (0, self.neg_ent)), mode='constant', constant_values=1)
			# batch_tp = np.pad(batch_tp, ((0, 0), (0, self.neg_ent)), mode='constant', constant_values=0)
      		

			batch_h = np.repeat(batch_h, 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_t = np.repeat(batch_t, 1 + self.neg_ent + self.neg_rel, axis = -1)
			batch_r = np.repeat(batch_r, 1 + self.neg_ent + self.neg_rel, axis = -1)			
			batch_tp = np.repeat(batch_tp, 1 + self.neg_ent + self.neg_rel, axis = -1)
   
			for index, item in enumerate(data):
				last = 1
				if self.neg_ent > 0:
					neg_head, neg_tail = self.__normal_TransE_batch(item[0], item[1], item[2], self.neg_ent)
					if len(neg_head) > 0:
						batch_h[index][last:last + len(neg_head)] = neg_head
						last += len(neg_head)
					if len(neg_tail) > 0:
						batch_t[index][last:last + len(neg_tail)] = neg_tail
						last += len(neg_tail)
					if last < self.neg_ent:
						batch_h[index][last:1 + self.neg_ent] =  np.ones(1 + self.neg_ent-last)
						batch_t[index][last:1 + self.neg_ent] = np.zeros(1 + self.neg_ent-last)
						last = 1 + self.neg_ent
				if self.neg_rel > 0:
					neg_rel = self.__rel_batch(item[0], item[1], item[2], self.neg_rel)
					batch_r[index][last:last + len(neg_rel)] = neg_rel
				# todo 是否生成时间字段的负样本
				if self.neg_time > 0:
					pass
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

		batch_y = np.concatenate([np.ones((len(data), 1)), np.zeros((len(data), self.neg_ent + self.neg_rel))], -1).transpose()
		batch_data['batch_h'] = batch_h.squeeze()
		batch_data['batch_t'] = batch_t.squeeze()
		batch_data['batch_r'] = batch_r.squeeze()
		batch_data['batch_tp'] = batch_tp.squeeze()
		batch_data['batch_y'] = batch_y.squeeze()
		return batch_data

	def __count_htr(self):
		"""统计训练集head、tail、relation数据
		"""
		self.h_of_tr = {}
		self.t_of_hr = {}
		self.r_of_ht = {}
		self.h_of_r = {}
		self.t_of_r = {}
		self.freqRel = {}
		self.lef_mean = {}
		self.rig_mean = {}

		triples = zip(self.all_head, self.all_tail, self.all_rel)
		for h, t, r in triples:
			if (h, r) not in self.t_of_hr:
				self.t_of_hr[(h, r)] = []
			self.t_of_hr[(h, r)].append(t)
			if (t, r) not in self.h_of_tr:
				self.h_of_tr[(t, r)] = []
			self.h_of_tr[(t, r)].append(h)
			if (h, t) not in self.r_of_ht:
				self.r_of_ht[(h, t)] = []
			self.r_of_ht[(h, t)].append(r)
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

	def __corrupt_head_norepeat(self, t, r):
		tmp = np.arange(self.ent_total)
		if not self.filter_flag or  (t, r) not in self.h_of_tr:
			return tmp
		mask = np.in1d(tmp, self.h_of_tr[(t, r)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_tail_norepeat(self, h, r):
		tmp = np.arange(self.ent_total)
		if not self.filter_flag or (h, r) not in self.t_of_hr:
			return tmp
		mask = np.in1d(tmp, self.t_of_hr[(h, r)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __corrupt_rel_norepeat(self, h, t):
		tmp = np.arange(self.ent_total)
		
		if not self.filter_flag or (h, t) not in self.r_of_ht:
			return tmp
		mask = np.in1d(tmp, self.r_of_ht[(h, t)], assume_unique=True, invert=True)
		neg = tmp[mask]
		return neg

	def __normal_batch(self, h, t, r, neg_size):
		neg_size_h = 0
		neg_size_t = 0
		# transH的负采样策略
		prob = self.rig_mean[r] / (self.rig_mean[r] + self.lef_mean[r]) if self.bern_flag else 0.5
		for i in range(neg_size):
			if random.random() < prob:
				neg_size_h += 1
			else:
				neg_size_t += 1

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

	def __normal_TransE_batch(self, h, t, r):
		"""
  		 取负样本的时候根据TransE的策略，取全量的数据
    	"""
		
		neg_list_h = []
		neg_list_t = []
  
		neg_tmp_h = self.__corrupt_head_norepeat(t, r)
		neg_tmp_h = neg_tmp_h[neg_tmp_h != h]
		neg_list_h.append(neg_tmp_h)
		
		neg_tmp_t = self.__corrupt_tail_norepeat(h, r)
		neg_tmp_t = neg_tmp_t[neg_tmp_t != t]
		neg_list_t.append(neg_tmp_t)
		
			
		if neg_list_h != []:
			neg_list_h = np.concatenate(neg_list_h)
			
   
		if neg_list_t != []:
			neg_list_t = np.concatenate(neg_list_t)

		return neg_list_h, neg_list_t

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
		return max(self.start_time)

	def get_tp_min(self):
		return min(self.start_time)

	def get_tp_tot(self):
		return self.get_tp_max() - self.get_tp_min() + 1
	

	

class PyTorchTestDataLoader(DataLoader):

	def __init__(self, 
		in_path = None, 
		tri_file = None,
		test_tri_file = None,
		threads = 8, 
		sampling_mode = "normal", 
		bern_flag = False, 
		filter_flag = True, 
		ent_tot=0,
		rel_tot=0,
		# neg_ent = 1, 
		# neg_rel = 0, 
		time_point_formate = "%Y",  # %Y year;  %Y-%m month;  %Y-%m day
		shuffle = True, 
		drop_last = True):

		self.st_dft = "1900"
		self.et_dft = "2024"

		self.in_path = in_path
		self.tri_file = tri_file
		self.test_tri_file = test_tri_file
		self.ent_tot = ent_tot
		self.rel_tot = rel_tot

		
		if in_path != None:
      
			self.test_tri_file = in_path + "test.txt"
			self.tri_file = in_path + "train2id.txt"
			
  
		self.time_point_type = time_point_formate
		dataset = self.__construct_dataset(sampling_mode, bern_flag, filter_flag, time_point_formate)

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
	
	#获取两个日期间的所有日期	
	def getEveryDay(begin_date,end_date):
		date_list = []
		begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
		end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
		while begin_date <= end_date:
			date_str = begin_date.strftime("%Y-%m-%d")
			date_list.append(date_str)
			begin_date += datetime.timedelta(days=1)
		return date_list

	def getEverYear(self, begin_date, end_date, date_formate):
		date_list = []
		date_list = list(range(int(begin_date), int(end_date)+1))
		return date_list
 		
	def get_time_point(self, h, r, t, st, et, date_formate):
		time_list = self.getEverYear(st, et, date_formate)
		h_tp, r_tp, t_tp, tp = [], [], [], []
		# assert len(h) == len(r) == len(t)
		# for i in range(len(h)):
		for time in time_list:
			h_tp.append(int(h))
			r_tp.append(int(r))
			t_tp.append(int(t))
			tp.append(int(time))
		return h_tp, r_tp, t_tp, tp					

	def __construct_dataset(self, sampling_mode, bern_flag, filter_flag, time_point_formate):
		
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
		# f = open(self.test_tri_file, "r")
		# test_triples_total = (int)(f.readline())
		with open(self.test_tri_file, "r") as f:
			for  line in f:
				# 注意需要与数据集文件每列的含义对齐
				# h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
				h, r, t, st, et = line.strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
				st = st.split("-")[0] # 只用年
				et = et.split("-")[0] # 只用年
				if st == "####" or len(st) != 4:
					st = self.st_dft
				if et == "####" or len(et) != 4:
					et = self.et_dft	
				
	
				h_tp, r_tp, t_tp, tp = self.get_time_point(h, r, t, st, et, time_point_formate)
				head_tp.extend(h_tp)
				rel_tp.extend(r_tp)
				tail_tp.extend(t_tp)
				tp_point.extend(tp)
  
	
						
		
  		# 加载训练数据
		f = open(self.tri_file, "r")
		triples_total = (int)(f.readline())
  
		for index in range(triples_total):
			# 注意需要与数据集文件每列的含义对齐
			# h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
			h, r, t, st, et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
			st = st.split("-")[0] # 只用年
			et = et.split("-")[0] # 只用年
			if st == "####" or len(st) != 4:
				st = self.st_dft
			if et == "####" or len(et) != 4:
				et = self.et_dft	
    
			
			all_head.append((int)(h))
			all_tail.append((int)(t))
			all_rel.append((int)(r))
			
			

			# 加入时间字段
			# start_time.append(st)
			# end_time.append(et)
		f.close()

		dataset = PyTorchTestDataset(np.array(head_tp), np.array(tail_tp), np.array(rel_tp), np.array(tp_point),
                               np.array(all_head), np.array(all_tail), np.array(all_rel), np.array(all_tp_point),
                               self.ent_tot,self.rel_tot,
                            sampling_mode, bern_flag, filter_flag)
  
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

	def get_tp_min(self):
    		return self.dataset.get_tp_min()


if __name__ == '__main__':
    year_formate = "%Y"
    month_formate = "%Y-%m"
    day_formte = "%Y-%m-%d"
    date_formate = month_formate
    date_list = PyTorchTestDataLoader.getEveryDay('2016-01','2019-03', date_formate)
    # print(sorted(set(date_list)))
    print(date_list)
