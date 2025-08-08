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
import copy
from tqdm import tqdm

# from openke.config.Tester_AddTime_tp_point import Tester_AddTime

class Trainer_AddTime(object):

    def __init__(self, 
                 model = None,
                 data_loader = None,
                 train_times = 1000,
                 alpha = 0.5,
                 use_gpu = True,
                 opt_method = "sgd"
                #  ,
                #  save_steps = None,
                #  checkpoint_dir = None,
                #  valid_data_loader_for_loss = None,
                #  valid_data_loader_for_test = None
     ):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        # self.save_steps = save_steps
        # self.checkpoint_dir = checkpoint_dir
        # self.valid_data_loader_for_loss = valid_data_loader_for_loss
        # self.valid_data_loader_for_test = valid_data_loader_for_test

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_tp': self.to_var(data['batch_tp'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()		 
        return loss.item()

    def valid_one_step(self, data):
        with torch.no_grad():
            loss = self.model({
                'batch_h': self.to_var(data['batch_h'], self.use_gpu),
                'batch_t': self.to_var(data['batch_t'], self.use_gpu),
                'batch_r': self.to_var(data['batch_r'], self.use_gpu),
                'batch_tp': self.to_var(data['batch_tp'], self.use_gpu),
                'batch_y': self.to_var(data['batch_y'], self.use_gpu),
                'mode': data['mode']
            })
        return loss.item() 
                 
    def valid_loss(self,epoch,valid_data_loader_for_loss):
        valid_res = 0.0
        num = 0
        for data in valid_data_loader_for_loss:
            valid_loss = self.valid_one_step(data)
            valid_res += valid_loss
            
            num+=1
        print("Epoch %d：valid loss is %f" %(epoch,valid_res/num))

    def run(self,tester = None,save_steps=None, checkpoint_dir = None,valid_data_loader_for_loss = None):
        if self.use_gpu:
            self.model.cuda()

        if self.optimizer != None:
            pass
        elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            self.optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.alpha,
                lr_decay=self.lr_decay,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            self.optimizer = optim.Adadelta(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.alpha,
                weight_decay=self.weight_decay,
            )
        else:
            print("self.model.parameters(): ",self.model.parameters())
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.alpha,
                weight_decay=self.weight_decay,
            )
        print("Finish initializing...")
        
        training_range = tqdm(range(self.train_times))
        for epoch in range(self.train_times):
            res = 0.0
            num=0
            for data in self.data_loader:
                loss = self.train_one_step(data)
               
                res += loss
                num+=1
                # curr_dim = len(data["batch_h"])
               
                # print(curr_dim)
               
                # count = curr_dim/11
                # print(count)
                # num+= count
            # training_range.set_description("Epoch %d | loss: %f" % (epoch, res/num))
            
            if save_steps and checkpoint_dir and (epoch + 1) % save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch+1))
                print("Epoch %d | train loss: %f" % (epoch+1, res/num))
                self.model.model.save_checkpoint(os.path.join(checkpoint_dir + "-" + str(epoch+1) + ".ckpt"))
                # 计算验证集的loss 
                self.valid_loss(epoch+1,valid_data_loader_for_loss)
                # 计算当前模型在验证集上的指标输出结果
                curr_model = self.model.model
                tester.set_model(curr_model)

                tester.run_link_prediction()
                


    def set_model(self, model):
        self.model = model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir = None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir