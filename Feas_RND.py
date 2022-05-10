"""
Author: Steve Paul 
Date: 4/28/22 """

from UAMEnv import UAMEnv
import torch as th
import numpy as np
import pickle
import argparse

def main():
    n_vertiports = 8  # number of vertiports
    n_evtols = 18  # number of evtols
    max_x_y_axis = 50
    vertiport_locations = th.rand((n_vertiports, 2)) * max_x_y_axis  # actual coordinates of the vertiports
    evtols_initial_locations = th.randint(0, n_vertiports, (n_evtols, 1))  # initial locations of the evtols
    demand = th.randint(0, 100, (n_vertiports, n_vertiports))  # constant demand for now
    ticket = demand * .05  # price of the ticket
    for i in range(n_vertiports):
        demand[i, i] = 0
    time_horizon = 18.00  # time before the last decision
    env = UAMEnv(
        n_vertiports=n_vertiports,
        n_evtols=n_evtols,
        vertiport_locations=vertiport_locations,
        evtols_initial_locations=evtols_initial_locations,
        demand_model=demand,
        time_horizon=time_horizon,
        max_x_y_axis=max_x_y_axis
    )

    obs = env.reset()
    reward_agg = []
    profit_agg = []
    method = "Feas_Rnd"
    dir = "Results/"
    data = {}
    test_envt_list = []
    with open("test_envts.pkl", 'rb') as f:
        test_data = pickle.load(f)
    test_envt_list = test_data["env"]
    first_obs_list = test_data["first_obs"]
    j = 0
    env = test_envt_list[j]
    obs = first_obs_list[j]
    env.traj_log_list = []
    for i in range(10000000):
        mask = obs["mask"]
        mask_non_zeros = mask.nonzero()[:, 0]

        ind = th.randint(0, mask_non_zeros.shape[0], (1, 1))
        action = mask_non_zeros[ind]
        # print(mask_non_zeros)
        # print(action[0])
        obs, reward, done, _ = env.step(action[0].item())
        if done:
            env.total_reward = env.total_reward-.02
            print(env.total_reward)
            reward_agg.append(env.total_reward)
            profit = env.total_reward * (
                (env.time_varying_demand_model["demand"] * env.passenger_pricing_model).sum()).item()
            profit_agg.append(profit)
            if len(reward_agg) == 100:
                data["total_reward"] = np.array(reward_agg)
                data["profit"] = np.array(profit_agg)
                with open(dir + method + "_results.pkl", 'wb') as f:
                    pickle.dump(data, f)
                print(np.array(reward_agg).mean(), np.array(reward_agg).std(), np.array(profit_agg).mean())
                break
            j = j + 1
            env = test_envt_list[j]
            env.traj_log_list = []
            obs = first_obs_list[j]




if __name__ == '__main__':
    main()
