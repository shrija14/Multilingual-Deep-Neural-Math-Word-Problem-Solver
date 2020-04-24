# encoding: utf-8
import os
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='word problems with seq2seq generating')

    '''数据加载配置'''
    '''
    parser.add_argument('--in-data-dir', type=str, dest='in_data_dir',\
                        default='./data/math23k_processed.json')
    parser.add_argument('--ex-1w-data-dir', type=str, dest='ex_1w_data_dir',
                        default='filter_external_template.json')
    parser.add_argument('--ex-5w7-data-dir', type=str, dest='ex_5w7_data_dir')
    parser.add_argument('--emb-dir', type=str, dest='emb_dir',default='')
    parser.add_argument('--token-list-dir', type=str, dest='token_list_dir',default='')
    '''

    parser.add_argument('--resume', action='store_true', dest='resume', default=False)
    parser.add_argument('--cuda-use', action='store_true', dest='cuda_use', default=False)
    parser.add_argument('--checkpoint-dir-name', type=str, dest='checkpoint_dir_name',\
                            default='0000_0000')
    parser.add_argument('--load-name', type=str, dest='load_name',default='best')
    parser.add_argument('--mode', type=int, dest='mode', default=0)
    parser.add_argument('--teacher-forcing-ratio', type=float, dest='teacher_forcing_ratio', default=1)
    parser.add_argument('--run-flag', type=str, dest='run_flag',default='train_23k')

    parser.add_argument('--encoder-hidden-size', type=int, dest='encoder_hidden_size',default=256)
    parser.add_argument('--decoder-hidden-size', type=int, dest='decoder_hidden_size',default=512)
    parser.add_argument('--dropout-p', type=float, dest='dropout_p',default=0.5)
    parser.add_argument('--input-dropout', type=float, dest='input_dropout',default=0)
    parser.add_argument('--layer', type=int, dest='layer',default=2)
    parser.add_argument('--cuda-id', type=str, dest='cuda_id',default='0')
    parser.add_argument('--generator', type=str, dest='generator',default=0)

    args = parser.parse_args()
    return args
