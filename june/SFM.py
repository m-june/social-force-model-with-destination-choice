import numpy as np
import math
import pandas as pd
from statistics import mean, stdev
from tqdm.notebook import trange, tqdm
import warnings
import argparse
from tqdm import tqdm
import time
import os
warnings.simplefilter('ignore')

from social_force_model_for_pcs import SocialForceModel

import sys
sys.path.append("../src")

from main import Agent, run
# from make_sfm import run_sfm
from destination_choice_model import DestinationChoiceModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='/home/aaf15257iq/work/GC_annotation/processed_data/duration_60_0.8/GC_Dataset_ped1-12685_time2400-2460_interp0_xrange5-25_yrange15-35.npy')
    parser.add_argument('-o', '--output', type=str, default='/home/aaf15257iq/work/GC_annotation/simulated_data/duration_60_0.8/')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('--a1', type=float, default=3.3)
    parser.add_argument('--b', type=float, default=0.4)
    parser.add_argument('--a2', type=float, default=1)
    parser.add_argument('--c', type=float, default=0.5) 
    args = parser.parse_args()
    return args

# Agent
class AgentSFM():
    def __init__(self, no, t0, r, v0, loc, vel, dest):
        self.no = no
        self.t0 = t0
        self.r = r
        self.v0 = v0
        self.loc = loc
        self.vel = vel
        self.dest = dest
        self.done = False

def run_SFM(trajectories, mean_v, mean_vx, mean_vy, num_step, sfm, dests):
     # initialize agents
    num_agent = len(trajectories)
    agents = {}
    for i in range(num_agent):
        agents[i] = AgentSFM(no=i,
                          t0=trajectories[i][0][-1],
                          r=0.2,
                          v0=np.array(mean_v[i]),
                          loc=[np.array([np.nan, np.nan, np.nan])],
                          vel=[np.array([np.nan, np.nan])],
                          dest=[np.array([np.nan, np.nan])],
                          )
    
    progress_bar = tqdm(total=num_step, desc="Simulation Progress")

    t = 0
    while t < num_step:  # until all pedestrians move to right
        for i in range(num_agent):
            # generate agents
            # if agents[i].t0 == int(t/10) and t % 10 == 0:
            if agents[i].t0 == t:
                agents[i].loc[t] = np.array(
                    [trajectories[i][0][0], trajectories[i][0][1], t])
                agents[i].vel[t] = np.array([np.mean(mean_vx), np.mean(mean_vy)])
                agents[i].dest[t] = dests[i][:2]
                # print(f"agents[{i}].dest[{t}]", agents[i].dest[t])

            # model
            if (agents[i].t0 <= t) and (agents[i].done == False):

                dest = agents[i].dest[t]

                loc_next, vel_next = sfm.step(t, i, agents, dest)

                agents[i].loc.append(loc_next)
                agents[i].vel.append(vel_next)
                agents[i].dest.append(dest)
                # print(loc_next)
                # print(vel_next)
                # print(dest)

                if np.linalg.norm(loc_next[:2] - dest) <= 0.1:
                    agents[i].done = True
            else:
                agents[i].loc.append(np.array([np.nan, np.nan, np.nan]))
                agents[i].vel.append(np.array([np.nan, np.nan]))
                agents[i].dest.append(np.array([np.nan, np.nan]))
        t += 1
        progress_bar.update(1)

    progress_bar.close()
    return agents

def read_trajectory(agents):
    print("read trajectory")
    # print(agents)
    trajectories = []
    for i in range(len(agents)):
        traj = agents[i].loc
        traj = [row.tolist() for row in traj if not np.isnan(row).any()]
        # print(traj)
        # tr = []
        # for t in range(0, len(traj), 10):
        #     time = int(t/10)
        #     tr.append(traj[t])
        #     tr[time][2] = int(traj[t][2]/10)
        # print('tr:',tr)
        # exit()
        traj = convert_last_to_int(traj)
        trajectories.append(traj)
    
    destinations = []
    for ag in trajectories:
        dest = ag[-1]
        destinations.append(dest)
    
    return trajectories, destinations

def convert_to_tuples(lst):
    return [tuple(item) for item in lst]

def convert_to_tuples2(lst):
    return [[tuple(item)] for item in lst]

def convert_last_to_int(lst):
    return [(item[0], item[1], int(item[2])) for item in lst]



# __all__ = ['runSFM']

if __name__ == "__main__":
    args = get_args()
    start_time = time.time()
    data_path = args.input
    load_path = data_path
    data = np.load(load_path, allow_pickle=True)
    meta_data, trajectories, destinations, obstacles = data
    
    # rewrite the meta_data to 0.8
    meta_data['time_unit'] = 0.8
    meta_data['interpolation'] = 0
    print(meta_data)
    # print(destinations)
    # exit()
    # trajectories_ = trajectories
    if len(trajectories) == 0:
        print("No trajectories")
        file_name = os.path.basename(load_path)
        file_name = file_name.split(".")[0]
        file_name = file_name + "_sfm.npy"
        save_path = os.path.join(args.output, file_name)
        data = np.array((meta_data, trajectories, destinations, obstacles), dtype=object)
        np.save(save_path, data)
        print("saved for ", save_path)
        print("Total time: ", time.time() - start_time)
        exit()
    mean_v = []
    for u in trajectories:
        tau = u[-1][-1] - u[0][-1]
        if tau >= 3:
            first = u[0][:2]
            last = u[3][:2]
            tau = 3
        else:
            first = u[0][:2]
            last = u[-1][:2]
            tau = u[-1][-1] - u[0][-1]
        first, last = np.array(first), np.array(last)
        v = (last - first) / tau
        v = v / 0.8
        v = v.tolist()
        mean_v.append(v)
    mean_vl = [np.linalg.norm(v) for v in mean_v]
    mean_vx = [v[0] for v in mean_v]
    mean_vx = [value for value in mean_vx if not np.isnan(value)]
    mean_vy = [v[1] for v in mean_v]
    mean_vy = [value for value in mean_vy if not np.isnan(value)]
    
    params_sfm = {
        "dt": 0.8,
        "A1": 3.3,
        "B": 0.4,
        "A2": 1,
        "tau": 0.5,
        "phi": 100,
        "c": 0.5
    }
    walls = np.array(obstacles)
    walls_points = walls.tolist()
    sfm = SocialForceModel(params_sfm, walls_points)
    num_steps = max([u[-1][-1] for u in trajectories]) + 1 #* 10
    # num_steps = 100 # for test
    # dests = [[x[0] for x in sublist] for sublist in destinations]
    dests = [list(item[0]) for item in destinations]
    # print(dests)

    agents = run_SFM(trajectories, mean_vl, mean_vx, mean_vy, num_steps, sfm, dests)

    trajectories, destinations = read_trajectory(agents)

    converted_trajectories = []
    for traj in trajectories:
        converted_traj = convert_to_tuples(traj)
        converted_trajectories.append(converted_traj)
    trajectories = converted_trajectories
    destinations = convert_to_tuples2(destinations)
    # print("destinations")
    # print(destinations)

    # file_path = os.path.dirname(load_path)
    file_name = os.path.basename(load_path)
    file_name = file_name.split(".")[0]
    file_name = file_name + "_sfm.npy"
    save_path = os.path.join(args.output, file_name)
    data = np.array((meta_data, trajectories, destinations, obstacles), dtype=object)
    np.save(save_path, data)
    print("saved for ", save_path)
    print("Total time: ", time.time() - start_time)
