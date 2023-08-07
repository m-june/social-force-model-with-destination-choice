#SFM.pyで作られたtrajectoriesと元のデータを比較する
#比較はmaeを用いる
#maeはMean Squared Errorの略で、二つのデータの差の二乗の平均を表す


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import warnings
import argparse
from tqdm import tqdm

warnings.simplefilter('ignore')




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--real', type=str, default='/home/aaf15257iq/work/equivariant-PIML/data/GC_dataset/GC_Dataset_ped1-12685_time2344-2404_interp9_xrange5-25_yrange15-35.npy')
    parser.add_argument('-s', '--sim', type=str, default='/home/aaf15257iq/work/equivariant-PIML/data/GC_dataset/GC_Dataset_ped1-12685_time2344-2404_interp9_xrange5-25_yrange15-35_sfm.npy')

    args = parser.parse_args()
    return args

#realとsimの加速度の差を計算する
def acc_mae():
    args = get_args()
    real = np.load(args.real, allow_pickle=True)
    sim = np.load(args.sim, allow_pickle=True)
    print(real.shape)
    print(sim.shape)
    real_trajectories = real[1]
    sim_trajectories = sim[1]


    progress_bar = tqdm(total=len(real_trajectories), desc="Function Progress")
    mae = 0
    mse = 0
    for i in range(len(real_trajectories)):
        first_step = real_trajectories[i][0][-1]
        num_step = min(real_trajectories[i][-1][-1], sim_trajectories[i][-1][-1]) + 1 - first_step
        t = 0
        while t < num_step - 2:
            real_vel = np.array(real_trajectories[i][t+1][:2]) - np.array(real_trajectories[i][t][:2])
            real_vel = real_vel/0.08
            real_vel_ = np.array(real_trajectories[i][t+2][:2]) - np.array(real_trajectories[i][t+1][:2])
            real_vel_ = real_vel_/0.08
            real_acc = (real_vel_ - real_vel)/0.08
            sim_vel = np.array(sim_trajectories[i][t+1][:2]) - np.array(sim_trajectories[i][t][:2])
            sim_vel = sim_vel/0.08
            sim_vel_ = np.array(sim_trajectories[i][t+2][:2]) - np.array(sim_trajectories[i][t+1][:2])
            sim_vel_ = sim_vel_/0.08
            sim_acc = (sim_vel_ - sim_vel)/0.08
            mae += np.linalg.norm(real_acc - sim_acc)
            mse += np.linalg.norm(real_acc - sim_acc) ** 2
            t += 1
        progress_bar.update(1)
    mae /= len(real_trajectories)
    mse /= len(real_trajectories)
    progress_bar.close()
    return mae, mse

#realとsimの位置の差を計算する
def posi_mae():
    args = get_args()
    real = np.load(args.real, allow_pickle=True)
    sim = np.load(args.sim, allow_pickle=True)
    print(real.shape)
    print(sim.shape)
    real_trajectories = real[1]
    sim_trajectories = sim[1]

    progress_bar = tqdm(total=len(real_trajectories), desc="Function Progress")
    mae = 0
    for i in range(len(real_trajectories)):
        m = 0
        first_step = real_trajectories[i][0][-1]
        num_step = min(real_trajectories[i][-1][-1], sim_trajectories[i][-1][-1]) + 1 - first_step
        t = 0
        while t < num_step:
            real_posi = np.array(real_trajectories[i][t][:2])
            sim_posi = np.array(sim_trajectories[i][t][:2])
            m += np.linalg.norm(real_posi - sim_posi)
            # mse += np.linalg.norm(real_acc - sim_acc) ** 2
            t += 1
        # m = m / num_step
        progress_bar.update(1)
        mae += m
    mae /= len(real_trajectories)
    progress_bar.close()
    return mae

if __name__ == '__main__':
    acc_mae, acc_mse = acc_mae()
    posi_mae = posi_mae()
    print("ACC_MAE=",acc_mae)
    print("ACC_MSE=",acc_mse)
    print("POS_MAE=",posi_mae)