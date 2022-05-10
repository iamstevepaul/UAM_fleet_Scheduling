"""
Author: Steve Paul 
Date: 4/29/22 """
from torch import nn
from collections import defaultdict
import warnings
import math
import numpy as np
import gym
from stable_baselines_al import PPO, A2C
# from stable_baselines.common import make_vec_env
from UAMEnv import UAMEnv
import json
import datetime as dt
import torch as th
from CustomNN import CustomNN
from Policies.CustomPolicies import ActorCriticGCAPSPolicy
import pickle

import scipy.sparse as sp
from persim import wasserstein, bottleneck
import ot
# from CustomPolicies import ActorCriticGCAPSPolicy
from stable_baselines_al.common.utils import set_random_seed


from stable_baselines_al.common.vec_env import DummyVecEnv, SubprocVecEnv

n_instances = 100
envt_list = []
first_obs_list = []
for i in range(n_instances):

    n_vertiports = 8 # number of vertiports
    n_evtols = 19 # number of evtols
    max_x_y_axis = 50
    vertiport_locations = th.rand((n_vertiports,2))*max_x_y_axis # actual coordinates of the vertiports
    evtols_initial_locations = th.randint(0, n_vertiports, (n_evtols, 1)) # initial locations of the evtols
    demand = th.randint(0, 100, (n_vertiports, n_vertiports)) # constant demand for now
    ticket = demand*.05 # price of the ticket
    for i in range(n_vertiports):
        demand[i,i] = 0
    time_horizon = 18.00 # time before the last decision
    env = UAMEnv(
            n_vertiports=n_vertiports,
            n_evtols=n_evtols,
            vertiport_locations=vertiport_locations,
            evtols_initial_locations = evtols_initial_locations,
            demand_model=demand,
            time_horizon = time_horizon,
            max_x_y_axis = max_x_y_axis
        )
    obs = env.reset()
    first_obs_list.append(obs)
    envt_list.append(env)
data = {
    "env": envt_list,
    "first_obs": first_obs_list
}
with open("test_envts_2.pkl", 'wb') as f:
    pickle.dump(data, f)
    f.close()