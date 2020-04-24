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

def main():
    if(len(sys.argv)>1):
        if(sys.argv[1]=="true"):
            predict = True
        else:
            predict = False
    else:
        predict = False
    if(predict):
        print("yes")
    else:
        print("no")
    dataset = read_data_json("./data/final_combined_replicate.json")
    count = 0
    max_l = 0
    if predict:
        norm_templates = dict(read_data_json("./data/pg_seq_norm_0821_test_best.json"))
    newd = {}
    for key, elem in dataset.items():
        if predict:
            if key in norm_templates:
                elem['post_template'] = norm_templates[key]
                elem['gd_tree_list'] = form_gdtree(elem)
                if(len(elem['gd_tree_list'])):
                    newd[key] = elem
        else:
            elem['gd_tree_list'] = form_gdtree(elem)
            if(len(elem['gd_tree_list'])):
                newd[key] = elem
    print("new", len(newd))
    data_loader = DataLoader(newd) 
    print ('loading finished')

    if os.path.isfile("./ablation_recursive-Copy1.log"):
        os.remove("./ablation_recursive-Copy1.log")

    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename='mylog.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.DEBUG)


    params = {"batch_size": 64, #64
              "start_epoch" : 1,
              "n_epoch": 20,	# 100, 50
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


    trainer = Trainer(data_loader, params, predict)
    if predict == True:
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
        model = torch.load('./data/model_combined_att_20.pt')
        trainer.predict_joint(model).to(device)

main()
