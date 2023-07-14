import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from tqdm.notebook import trange, tqdm
import warnings
import argparse
import torch
import time
from tqdm import tqdm
import yaml
import os
import subprocess
import argparse
warnings.simplefilter('ignore')

import sys
sys.path.append("../src")

from main import Agent, run
# from make_sfm import run_sfm
from social_force_model import SocialForceModel
from destination_choice_model import DestinationChoiceModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config.yaml')
    parser.add_argument('--dataset_flag', type=bool, default=True)
    parser.add_argument('-a', '--abci_flag', type=bool, default=False)
    parser.add_argument('-d', '--device', type=str, default='cpu')  
    args = parser.parse_args()
    return args

#yamlから読み込むデータパスを指定
#データパスを使ってmake_sfmを実行

if __name__ == '__main__':
    args = get_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if args.dataset_flag:
        print('dataset_flag is True')
        path = config['input']

        # パスにあるファイルの取得
        file_list = []
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                file_list.append(file_path)

        # ファイルリストの表示
        for path in file_list:
            print(path)
        data_path = file_list
        save_path = config['output']
        device = config['device']

    else:
        data_path = config['input']
        save_path = config['output']
        device = config['device']
    if args.abci_flag:
        print('abci_flag is True')
        for i in range(len(data_path)):
            dp = data_path[i]
            sp = save_path
            command = 'qsub -g gaa50073 /home/aaf15257iq/script/job_script/make_trajectory.sh ' + dp + ' ' + sp + ' ' + device
            print(command)
            command = command.split()
            run = subprocess.call(command)
            print('command number:', i)
            if run == 0:
                print('success')
            else:
                print('failed')
                break
    
    else:
        print('abci_flag is False')
        for i in range(len(data_path)):
            dp = data_path[i]
            sp = save_path
            command = 'python SFM.py -i ' + dp + ' -o ' + sp + ' -d ' + device
            print(command)
            command = command.split()
            run = subprocess.call(command)
            print('command number:', i)
            if run == 0:
                print('success')
            else:
                print('failed')
                break
    print('finish')
    