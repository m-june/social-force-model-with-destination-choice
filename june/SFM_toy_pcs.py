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
    parser.add_argument('-i', '--input', type=str, default='/homes/jmiwa/miwa/work/GC_data/processed_data/duration_60/GC_Dataset_ped1-12685_time0-60_interp9_xrange5-25_yrange15-35.npy')
    parser.add_argument('-o', '--output', type=str, default='/homes/jmiwa/miwa/work/GC_data/simulated_data/toy')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('--file_name', type=str, default='toy')
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

                vel_cur = agents[i].vel[t]
                loc_cur = agents[i].loc[t]

                loc_next, vel_next = sfm.step(t, i, agents, dest)

                # if vel_cur[0] != vel_next[0]:
                #     print(vel_cur, vel_next, t, i)
                #     break
                # print('val', vel_cur == vel_next)
                # print('loc', loc_cur, loc_next)

                agents[i].loc.append(loc_next)
                agents[i].vel.append(vel_next)
                agents[i].dest.append(dest)
                # print(loc_next)

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
        print('traj',traj[:50])
        # exit()
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

#trajectoryから速度を計算→agentsの速度と比較
def check_velocity(trajectories, agents, dt=0.08):
    for i in range(len(trajectories)):
        traj = trajectories[i]
        agent = agents[i]
        count = 0
        # for t in range(len(traj)-1):
        for t in range(29):
            loc1 = traj[t]
            loc2 = traj[t+1]
            loc3 = traj[t+2]
            print(agent.loc[t]==loc1)
            print(loc1,loc2)
            v_l = np.array([loc2[0]-loc1[0], loc2[1]-loc1[1]])
            v_l_2 = np.array([loc3[0]-loc2[0], loc3[1]-loc2[1]])
            v_1, v_2 = v_l / dt, v_l_2 / dt
            a_l = v_2 - v_1
            a = a_l / dt
            print('a', a, 't', t)
            # v = np.round(v, 8)
            # print(v == agent.vel[t])
            # print(v)
            # print(agent.vel[t])
            # if not np.allclose(v, agent.vel[t]):
            #     print("velocity is different")
            #     print(v, agent.vel[t])
            #     print("time:", t)
            #     print("agent:", i)
            #     print("trajectory:", traj)
            #     print("agent velocity:", agent.vel)
            #     print("trajectory velocity:", [np.linalg.norm([traj[j+1][0]-traj[j][0], traj[j+1][1]-traj[j][1]]) for j in range(len(traj)-1)])
            #     count += 1
            #     if count > 5:
            #         print("too many errors")
            #         print(t)
            #         break


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
    print('meta_data', meta_data)
    
    #自分でエージェントを設定する場合
    #2つのエージェントを設定
    trajectories = [
        [[5, 25, 0]
        ],
        [[14, 35, 0]
        ]
    ]

    dests = [
        [25, 25],
        [15, 15]
    ]

    mean_v = [[1.5, 0], [0, -1.5]]
    mean_vl = [np.linalg.norm(v) for v in mean_v]
    mean_vx = [v[0] for v in mean_v if not np.isnan(v[0])]
    mean_vy = [v[1] for v in mean_v if not np.isnan(v[1])]
    
    params_sfm = {
        "dt": 0.08,
        "A1": 5,
        "B": 1.0,
        "A2": 1,
        "tau": 0.5,
        "phi": 120,
        "c": 1.0
    }
    walls_points = []
    sfm = SocialForceModel(params_sfm, walls_points)
    # num_steps = max([u[-1][-1] for u in trajectories]) + 1 #* 10
    num_steps = 300 # for test
    # dests = [[x[0] for x in sublist] for sublist in destinations]

    agents = run_SFM(trajectories, mean_vl, mean_vx, mean_vy, num_steps, sfm, dests)

    new_trajectories, new_destinations = read_trajectory(agents)
    # print("new_destinations", new_destinations)
    # exit()
    new_destinations = [(25, 25, 167), (15, 15, 180)]

    check_velocity(new_trajectories, agents)

    converted_trajectories = []
    for traj in new_trajectories:
        converted_traj = convert_to_tuples(traj)
        converted_trajectories.append(converted_traj)
    new_trajectories = converted_trajectories
    new_destinations = convert_to_tuples2(new_destinations)
    new_obstacles = []

    # file_path = os.path.dirname(load_path)
    # file_name = os.path.basename(load_path)
    # file_name = file_name.split(".")[0]
    # file_name = file_name + "_sfm.npy"
    # save_path = os.path.join(args.output, file_name)
    # exit()
    file = args.file_name + "_sfm.npy"
    save_path = os.path.join(args.output, file)
    data = np.array((meta_data, new_trajectories, new_destinations, new_obstacles), dtype=object)
    np.save(save_path, data)
    print("saved for ", save_path)
    print("Total time: ", time.time() - start_time)
