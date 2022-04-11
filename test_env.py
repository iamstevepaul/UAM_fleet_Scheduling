"""
Author: Steve Paul 
Date: 3/9/22 """
from UAMEnv import UAMEnv
import torch as th
import numpy as np
import argparse

def main():
    x_min = 0
    x_max = 100
    y_min = 0
    y_max = 100
    n_vertiports = 10
    n_evtols = 13
    vertiport_locations = th.rand((n_vertiports,2))*100
    evtols_initial_locations = th.randint(0, n_vertiports, (n_evtols, 1))
    demand = th.randint(0, 100, (n_vertiports, n_vertiports))
    ticket = demand*.05
    for i in range(n_vertiports):
        demand[i,i] = 0
    time_horizon = 18.00
    env = UAMEnv(
        n_vertiports=n_vertiports,
        n_evtols=n_evtols,
        vertiport_locations=vertiport_locations,
        evtols_initial_locations = evtols_initial_locations,
        demand_model=demand,
        time_horizon = time_horizon,
        passenger_pricing_model=ticket
    )

    env.reset()
    for i in range(100000):
        mask = env.get_mask()
        mask_non_zeros = mask.nonzero()[:,0]
        ind = th.randint(0, mask_non_zeros.shape[0], (1,1))
        action = mask_non_zeros[ind]
        obs, reward, done, _ = env.step(action[0])
        if done:
            env.reset()





if __name__ == '__main__':
    main()
