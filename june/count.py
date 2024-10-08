import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from tqdm.notebook import trange, tqdm
import warnings
import argparse
import time
import datetime
from tqdm import tqdm
import yaml
import os
import subprocess
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
warnings.simplefilter('ignore')


def make_hist():
    data_path = '/home/aaf15257iq/work/GC_annotation/processed_data/GC_Dataset_ped1-12685_time0-3600_interp9_xrange5-25_yrange15-35.npy'
    # for data_path in data_paths:
    data = np.load(data_path, allow_pickle=True)
    ped_num = len(data[1])
    print('ped_num',ped_num)

    #pedの数だけループ
    most_long = 0
    long_num = 0
    ped_long_list = []
    for i in range(ped_num):
        ped_long = len(data[1][i])
        ped_long = ped_long * 0.08
        if ped_long > most_long:
            most_long = ped_long
        if ped_long > 60:
            long_num += 1
        ped_long_list.append(ped_long)
    print('most_long',most_long)
    print('long_num',long_num)

    plt.hist(ped_long_list, bins=150, alpha=0.7, color='blue')
    plt.xlabel('Pedestrian Duration (seconds)')
    plt.ylabel('Frequency')
    plt.xlim(0, 60) # x軸の範囲
    plt.title('Distribution of Pedestrian Durations')
    plt.show()

    # ヒストグラムを保存
    output_path = 'histogram_2.png'
    plt.savefig(output_path)

    # ヒストグラムを表示
    plt.show()

def count_yaml(config_path):
    with open(config_path, 'r') as stream:
        data_paths = yaml.load(stream, Loader=yaml.FullLoader)
    print(data_paths)
    data = defaultdict(list)
    for key in data_paths.keys():
        ped_num = 0
        for path in data_paths[key]:
            d = np.load(path, allow_pickle=True)
            ped_num += len(d[1])
        data[key] = ped_num
    print(data)
    return data, data_paths

def count_datanum(data_paths):
    with open(config_path, 'r') as stream:
        data_paths = yaml.load(stream, Loader=yaml.FullLoader)
    print(data_paths)
    data = defaultdict(list)
    for key in data_paths.keys():
        data_num = 0
        for path in data_paths[key]:
            d = np.load(path, allow_pickle=True)
            for i in range(len(d[1])):
                data_num += len(d[1][i])
        data[key] = data_num
    print(data)
    return data, data_paths

def save_dict_to_text(data, data_paths, save_path):
    with open(save_path, 'w') as f:
        f.write("Data Count:\n")
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

        f.write("\nData Paths:\n")
        for key, paths in data_paths.items():
            f.write(f"{key}:\n")
            for path in paths:
                f.write(f"  - {path}\n")

if __name__ == '__main__':
    # make_hist()
    config_path = '/home/aaf15257iq/work/equivariant-PIML/src/configs/data_configs/data_pretrain2.yaml'
    # data, paths = count_yaml(config_path)
    data, paths = count_datanum(config_path)
    config = os.path.basename(config_path).split('.')[0] 
    date = datetime.datetime.now().strftime('%m%d')
    save_path = 'text/' + str(config) + date + '.txt'
    save_dict_to_text(data, paths, save_path)
    print('finish')