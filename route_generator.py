"""
Author: Steve Paul 
Date: 5/1/22 """
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors
import torch as th


def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]

    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)


def view_colormap(cmap):
    """Plot a colormap with its grayscale equivalent"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))

    fig, ax = plt.subplots(2, figsize=(6, 2),
                           subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow([colors], extent=[0, 10, 0, 1])
    ax[1].imshow([grayscale], extent=[0, 10, 0, 1])

with open("logging_2.pkl", "rb") as f:

    traj_data = pickle.load(f)
    f.close()
vertiport_locations = traj_data["location"]
time_varying_demand_time = traj_data["time_varying_demand"]["time_points"]
time_varying_demand = traj_data["time_varying_demand"]["demand"]
trajectory = traj_data["trajectory"]
sizes = [4000]*8
demand_high = 110
ax = sns.scatterplot(vertiport_locations[:,0], vertiport_locations[:,1], s=500)

for i in range(len(traj_data["time_varying_demand"]["time_points"])):
    demand_mat = time_varying_demand[i,:,:]
    print(demand_mat)
    # ax = sns.scatterplot(vertiport_locations[:, 0], vertiport_locations[:, 1], s=500)
    for j in range(8):
        for k in range(8):

            if j != k:
                demand = demand_mat[j,k]
                x1 = vertiport_locations[j,0]
                x2 = vertiport_locations[k,0]
                y1 = vertiport_locations[j,1]
                y2 = vertiport_locations[k,1]
                dx = x2-x1
                dy=y2-y1
                # plt.arrow(x1, y1, dx, dy, width=.05,head_width = .3, length_includes_head=True, shape='left')
    trip_log = []
    trip_sum_mat = np.zeros((8,8))
    for traj in trajectory:
        time = traj[3].item()
        agent = traj[0]
        loc_1 = traj[1]
        loc_2 = traj[2]
        trip = traj[4]
        if time >= time_varying_demand_time[i] and time <= time_varying_demand_time[i+1]:
            if trip >= 0:
                trip_sum_mat[loc_1,loc_2] = trip_sum_mat[loc_1,loc_2] + 1
    colorarray=[]
    fig, (ax1, ax2) = plt.subplots(2)
    for j in range(8):
        for k in range(8):

            if j != k:
                if j < k:
                    n_tranported_high = 5*4
                    n_tranported = trip_sum_mat[j,k]*4
                    x1 = vertiport_locations[j,0]
                    x2 = vertiport_locations[k,0]
                    y1 = vertiport_locations[j,1]
                    y2 = vertiport_locations[k,1]
                    dx = x2-x1
                    dy=y2-y1
                    frac = (n_tranported/n_tranported_high).item()
                    fc = (frac, 1-frac, frac)
                    colorarray.append(fc)
                    ax1.arrow(x1, y1, dx, dy, width=frac/3+.1,head_width = .5, length_includes_head=True, shape='full', facecolor=fc, edgecolor=fc)

                    demand = demand_mat[j,k]
                    x1 = vertiport_locations[j,0]
                    x2 = vertiport_locations[k,0]
                    y1 = vertiport_locations[j,1]
                    y2 = vertiport_locations[k,1]
                    dx = x2-x1
                    dy=y2-y1
                    frac = (demand/demand_high).item()
                    fc = (frac, 1-frac, 0.3)
                    colorarray.append(fc)
                    ax2.arrow(x1, y1, dx, dy, width=.1,head_width = .5, length_includes_head=True, shape='full', facecolor=fc, edgecolor=fc)



    # cmap = ListedColormap(colorarray)
    # view_colormap(cmap)
    plt.show()
    # for j in range(8):
    #     for k in range(8):
    #
    #         if j != k:
    #             if j > k:
    #
    #                 demand = demand_mat[j,k]
    #                 x1 = vertiport_locations[j,0]
    #                 x2 = vertiport_locations[k,0]
    #                 y1 = vertiport_locations[j,1]
    #                 y2 = vertiport_locations[k,1]
    #                 dx = x2-x1
    #                 dy=y2-y1
    #                 frac = (demand/demand_high).item()
    #                 fc = (frac, 1-frac, 0.3)
    #                 colorarray.append(fc)
    #                 # plt.arrow(x1, y1, dx, dy, width=.1,head_width = .3, length_includes_head=True, shape='full', facecolor=fc, edgecolor=fc)
    #

    # cmap = ListedColormap(colorarray)
    # view_colormap(cmap)
    # plt.show()
    # ax = sns.scatterplot(vertiport_locations[:, 0], vertiport_locations[:, 1], s=500)
    # for j in range(8):
    #     for k in range(8):
    #
    #         if j != k:
    #             if j < k:
    #                 demand = demand_mat[j, k]
    #                 x1 = vertiport_locations[j, 0]
    #                 x2 = vertiport_locations[k, 0]
    #                 y1 = vertiport_locations[j, 1]
    #                 y2 = vertiport_locations[k, 1]
    #                 dx = x2 - x1
    #                 dy = y2 - y1
    #                 frac = (demand / demand_high).item()
    #                 fc = (frac, 1 - frac, 0.3)
    #                 colorarray.append(fc)
    #                 plt.arrow(x1, y1, dx, dy, width=.1, head_width=.3, length_includes_head=True, shape='full',
    #                           facecolor=fc, edgecolor=fc)
    #
    # # cmap = ListedColormap(colorarray)
    # # view_colormap(cmap)
    # plt.show()
    ft = 0

