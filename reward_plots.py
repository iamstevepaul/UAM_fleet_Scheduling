"""
Author: Steve Paul 
Date: 4/26/22 """

import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sb

linewidth = 4.0
x_lable_font_size = 40
y_lable_font_size = 40

x_tick_font_size = 40
y_tick_font_size = 40
legends_font_Size = 40
legends_line_width = 6
## 13 bus system
plt.figure()
bus = 13
methods = ["MLP", "GCAPS/PPO_17"]
data_13 = {}
for method in methods:
    if method != "MLP":
        flname = "GCAPS"
    else:
        flname = method+"2"
    file_name = 'logger/Trained/'+method+'/'+flname+'.csv'
    with open(file_name, newline='') as csvfile:
        datareader = csv.DictReader(csvfile)
        dm = []
        for row in datareader:
            dm.append([float(row["Value"]), int(row["Step"])])
    csvfile.close()
    data_13[flname] = np.array(dm)

plt.plot(data_13["GCAPS"][:99,1], data_13["GCAPS"][:99,0], data_13["MLP2"][:99,1], data_13["MLP2"][:99,0], linewidth=linewidth)
plt.xlabel('Steps',fontsize=x_lable_font_size)
plt.ylabel('Mean reward per episode', fontsize=y_lable_font_size)
plt.xticks(fontsize=x_tick_font_size)
plt.yticks(fontsize=y_tick_font_size)
leg = plt.legend(['GCAPS-RL', 'MLP-RL'], fontsize=legends_font_Size)
for legobj in leg.legendHandles:
    legobj.set_linewidth(legends_line_width)
plt.show()

plt.figure()
X_Y_Spline = make_interp_spline(data_13["GCAPS"][:,1], data_13["GCAPS"][:,0])
X_13_GCAPS = np.linspace(data_13["GCAPS"][:,1].min(), data_13["GCAPS"][:,1].max(), 10)
Y_13_GCAPS = X_Y_Spline(X_13_GCAPS)
X_Y_Spline = make_interp_spline(data_13["MLP"][:,1], data_13["MLP"][:,0])
X_13_MLP = np.linspace(data_13["MLP"][:,1].min(), data_13["MLP"][:,1].max(), 10)
Y_13_MLP = X_Y_Spline(X_13_MLP)
plt.plot(X_13_GCAPS, Y_13_GCAPS, X_13_MLP, Y_13_MLP, linewidth=linewidth)
plt.xlabel('Steps',fontsize=x_lable_font_size)
plt.ylabel('Mean reward per episode', fontsize=y_lable_font_size)
plt.xticks(fontsize=x_tick_font_size)
plt.yticks(fontsize=y_tick_font_size)
leg = plt.legend(['GCAPS-RL', 'MLP-RL'], fontsize=legends_font_Size)
for legobj in leg.legendHandles:
    legobj.set_linewidth(legends_line_width)
plt.show()


# plt.figure()
# bus = 34
# methods = ["MLP", "GCAPS"]
# data_34 = {}
# for method in methods:
#     file_name = str(bus)+'/'+method+'/'+str(bus)+'_'+method+'_rewards.csv'
#     with open(file_name, newline='') as csvfile:
#         datareader = csv.DictReader(csvfile)
#         dm = []
#         for row in datareader:
#             dm.append([float(row["Value"]), int(row["Step"])])
#     csvfile.close()
#     data_34[method] = np.array(dm)
#
# plt.plot(data_34["GCAPS"][:,1], data_34["GCAPS"][:,0], data_34["MLP"][:,1], data_34["MLP"][:,0], linewidth=linewidth)
# plt.xlabel('Steps',fontsize=x_lable_font_size)
# plt.ylabel('Mean reward per episode', fontsize=y_lable_font_size)
# plt.xticks(fontsize=x_tick_font_size)
# plt.yticks(fontsize=y_tick_font_size)
# leg = plt.legend(['GCAPS-RL', 'MLP-RL'], fontsize=legends_font_Size)
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(legends_line_width)
# plt.show()