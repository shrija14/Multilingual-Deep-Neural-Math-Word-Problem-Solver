import json                                                                                                                                                                                                                                                                   
import numpy as np
import time
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import logging
import random
import pickle
import sys

import os

from trainer import *
from utils import *
from model import *
from data_loader import *


import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#np.random.seed(123)
#random.seed(123)

def main():
    predict = False
    # dataset = read_data_json("./data/final_dolphin_data_replicate.json")
    # dataset = read_data_json("./data/final_combined_data_replicate.json")/
    dataset = read_data_json("./data/final_combined_replicate.json")


    #emb_vectors = np.load('./data/emb_100.npy')
    #dict_keys(['text', 'ans', 'mid_template', 'num_position', 'post_template', 'num_list', \
    #'template_text', 'expression', 'numtemp_order', 'index', 'gd_tree_list'])
    count = 0
    max_l = 0
    if predict:
        norm_templates = dict(read_data_json("./data/pg_seq_norm_0821_test.json"))
        # print(norm_templates)
        # norm_templates = {words[0]:words[1:][0] for words in norm_templates}
    # print(norm_templates)
    newd = {}
    for key, elem in dataset.items():
        #print (elem['post_template'])
        #print (norm_templates[key])
        #print ()
        if predict:
            if key in norm_templates:
                elem['post_template'] = norm_templates[key]
                elem['gd_tree_list'] = form_gdtree(elem)
                if(len(elem['gd_tree_list'])):
                    newd[key] = elem
        # print(key,elem)
        else:
            elem['gd_tree_list'] = form_gdtree(elem)
            # print("Length is ",len(elem['gd_tree_list']))
            if(len(elem['gd_tree_list'])):
                newd[key] = elem
            # print("hi")
            # print(dataset)
            # print(key,elem)
        # if len(elem['gd_tree_list']):
        #     #print (elem.keys())
        #     #print (elem['text'])
        #     #print (elem['mid_template'])
        #     #print (elem['post_template'])
        #     #print (elem['post_template'][2:])
        #     # l = max([int(i.split('_')[1]) for i in set(elem['post_template']) if 'temp' in i])
        #     # if max_l < l:
        #     #     max_l = l
        #     count += 1
    # print ("Max length of equation is", max_l)
    #print (elem['gd_tree_list'])
    # print (count)
    # print("poooo")
    print("new", len(newd))
    data_loader = DataLoader(newd) 
    # print("klpo")
    # print(newd)
    print ('loading finished')

    if os.path.isfile("./ablation_recursive-Copy1.log"):
        os.remove("./ablation_recursive-Copy1.log")

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='mylog.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)


    params = {"batch_size": 128, #64
              "start_epoch" : 1,
              "n_epoch": 30,	# 100, 50
              "rnn_classes":5,	# 5, 4
              "save_file": "model_combined_att.pt"
             }
    encode_params = {
        "emb_size":  512,
        "hidden_size": 160,	# 160, 128
        "input_dropout_p": 0.2,
        "dropout_p": 0.5,
        "n_layers": 2,
        "bidirectional": True,
        "rnn_cell": None,
        "rnn_cell_name": 'lstm',
        "variable_lengths_flag": True
    }

    #dataset = read_data_json("/home/wanglei/aaai_2019/pointer_math_dqn/dataset/source_2/math23k_final.json")
    #emb_vectors = np.load('/home/wanglei/aaai_2019/parsing_for_mwp/data/source_2/emb_100.npy')
    #data_loader = DataLoader(dataset) 
    #print ('loading finished')


    
    trainer = Trainer(data_loader, params, predict)

    if predict == False:
        recu_nn = RecursiveNN(data_loader.vocab_len, encode_params['emb_size'], params["rnn_classes"])
        #recu_nn = recu_nn.cuda()
        recu_nn = recu_nn.to(device)
        self_att_recu_tree = Self_ATT_RTree(data_loader, encode_params, recu_nn)
        #self_att_recu_tree = self_att_recu_tree.cuda()
        self_att_recu_tree = self_att_recu_tree.to(device)
        #for name, params in self_att_recu_tree.named_children():
        #    print (name, params)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self_att_recu_tree.parameters()), \
                                    lr=0.01, momentum=0.9, dampening=0.0)
        trainer.train(self_att_recu_tree, optimizer)
    else:
        model = torch.load('model_combined_att.pt')
        trainer.predict_joint(model).to(device)

main()
