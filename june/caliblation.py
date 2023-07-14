#SFM.pyで作られたtrajectoriesと元のデータを比較する
#比較はMSEを用いる
#MSEはMean Squared Errorの略で、二つのデータの差の二乗の平均を表す


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
    parser.add_argument('-r', '--real', type=str, default='/home/aaf15257iq/work/GC_annotation/processed_data/duration_60/GC_Dataset_ped1-12685_time2340-2400_interp9_xrange5-25_yrange15-35.npy')
    parser.add_argument('-s', '--sim', type=str, default='/home/aaf15257iq/work/GC_annotation/simulated_data/duration_60/GC_Dataset_ped1-12685_time2340-2400_interp9_xrange5-25_yrange15-35_sfm.npy')

    args = parser.parse_args()
    return args

#realとsimの加速度の差を計算する
def loss_function():
    args = get_args()
    real = np.load(args.real, allow_pickle=True)
    sim = np.load(args.sim, allow_pickle=True)
    print(real.shape)
    print(sim.shape)
    real_trajectories = real[1]
    sim_trajectories = sim[1]

    progress_bar = tqdm(total=len(real_trajectories), desc="Function Progress")
    mse = 0
    for i in range(len(real_trajectories)):
        first_step = real_trajectories[i][0][-1]
        num_step = min(real_trajectories[i][-1][-1], sim_trajectories[i][-1][-1]) + 1 - first_step
        t = 0
        while t < num_step - 2:
            real_vel = np.array(real_trajectories[i][t+1][:2]) - np.array(real_trajectories[i][t][:2])
            real_vel_ = np.array(real_trajectories[i][t+2][:2]) - np.array(real_trajectories[i][t+1][:2])
            real_acc = real_vel_ - real_vel
            sim_vel = np.array(sim_trajectories[i][t+1][:2]) - np.array(sim_trajectories[i][t][:2])
            sim_vel_ = np.array(sim_trajectories[i][t+2][:2]) - np.array(sim_trajectories[i][t+1][:2])
            sim_acc = sim_vel_ - sim_vel
            mse += np.linalg.norm(real_acc - sim_acc)
            # mse += np.linalg.norm(real_acc - sim_acc) ** 2
            t += 1
        progress_bar.update(1)
    mse /= len(real_trajectories)
    progress_bar.close()
    return mse

if __name__ == '__main__':
    mae = loss_function()
    print(mae)