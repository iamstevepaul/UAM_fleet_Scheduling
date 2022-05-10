"""
Author: Steve Paul 
Date: 4/27/22 """

from gurobipy import *
import random
import numpy as np
from matplotlib import pyplot as plt
import time
import itertools

start = time.time()

'''Problem parameter setup'''

# test case
# n=100, m=[25,20,15]
n_vertiports = 3
n_evtols = 6
# set of all vertiports
N = list(range(0, n_vertiports))

# set of all arcs
A = [[i,j] for i in N for j in N]
time_len = 27 #(6.0, 6.5 ....18.00)
# set of time steps
T = list(range(0, time_len))

max_xy = 3
vertiport_locations = [[random.randint(0,max_xy), random.randint(0,max_xy)] for i in range(n_vertiports)]
distance_matrix = [[np.linalg.norm((np.array(p1) - np.array(p2))) for p1 in vertiport_locations] for p2 in vertiport_locations]

# set of evtols
K = list(range(0, n_evtols)) # evtols
N_dash = [[n, t] for t in T for n in N]
Nd_len = len(N_dash)
#N_s set of all sapce time arc
A_dash = [] # set of all space time arcs
for n_d1 in N_dash:
    for n_d2 in N_dash:
        if n_d2[1] - n_d1[1] == 1:
            A_dash.append([n_d1,n_d2])
Ad_len = len(A_dash)
q = [random.randint(20,40) if ad[0][0] != ad[1][0] else 0 for ad in A_dash] #  potential demands in each arc - this will be changed
# e = [[random.randint(20,200) if it1 != it2 else 0 for it1 in N] for it2 in N] # energe required in each ar -  this will be changed
Emax = 110 # max battery capacity of the evtols
P_max = 150
evtol_max_range = 50 # miles
evtol_energy_per_mile = Emax/evtol_max_range
# c = [[random.randint(10,50) if it1 != it2 else 0 for it1 in N] for it2 in N] # cost of operating, this will not be used now, but needs to be cnhanged
C = 4 # passenger capacity of evtols
u = [6 for i in N]
p = np.array(distance_matrix)*1.5#np.array(c) + np.array(q) # fare charged to the passengers, this will be changed
d = q
E = list(np.array([distance_matrix for ti in range(Ad_len)])*evtol_energy_per_mile) # this should be checked for energy between same nodes
gamma = 0.1

LMPR = 0.2

delta_t = 1.0


model = Model("UAM fleet scheduling")

''' Decision Variables'''
xij = model.addVars(Ad_len, n_evtols, vtype=GRB.BINARY, name="xij")
xi = model.addVars(n_vertiports, n_evtols, vtype=GRB.BINARY, name="xi")
w = model.addVars(Ad_len, vtype=GRB.INTEGER, name="w")
# Cij = model.addVars(Ad_len, vtype=GRB.INTEGER, name="Cij")
Ei = model.addVars(Nd_len, n_evtols, vtype=GRB.CONTINUOUS, name="Ei")
# Eij = model.addVars(Ad_len, n_evtols, vtype=GRB.CONTINUOUS, name="Eij")
g = model.addVars(Ad_len, n_evtols, vtype=GRB.CONTINUOUS, name="g")

# constraints 6, 7 and 8
for ij in range(Ad_len):
    # model.addConstr(Cij[ij] == quicksum([C*xij[ij, k] for k in range(n_evtols)]))
    model.addConstr(w[ij] <= d[ij])  # 7
    model.addConstr(w[ij] <= quicksum([C*xij[ij, k] for k in range(n_evtols)])) # 6 and 8 combimned

for k in range(n_evtols):
    model.addConstr(quicksum([xi[i,k] for i in range(n_vertiports)]) == 1) # constraint 9

    for i in range(n_vertiports):
        # cosntraint 10 - check again
        model.addConstr(xi[i,k] ==  quicksum([xij[ij, k] if A_dash[ij][0] == [i, 0] else 0 for ij in range(Ad_len)])) # or Ndlen

# constriant 11
for k in range(n_evtols):

    for h in range(Nd_len):
        h_d = N_dash[h]
        if h_d[1] >= 1 and h_d[1] <= time_len -2: # or 2
            # or Ndlen
            model.addConstr(
                quicksum([xij[ij,k] if (A_dash[ij][1] == h_d) else 0 for ij in range(Ad_len)]) \
            == quicksum([xij[ij,k] if (A_dash[ij][0] == h_d) else 0 for ij in range(Ad_len)])
            )


# constraint 12
for i in range(n_vertiports):
    model.addConstr(quicksum(xi[i,k] for k in range(n_evtols)) <= u[i])

# constraint 13
for i in range(Nd_len):
    nd = N_dash[i] # double check the u part
    if nd[1] >= 1:
        model.addConstr(quicksum([xij[ij, k] if A_dash[ij][1] == nd else 0 for ij in range(Ad_len)  for k in range(n_evtols)]) <= u[nd[0]])


for i in range(Nd_len):
    for k in range(n_evtols):
        # constraints 14
        model.addConstr(Ei[i,k] >= 0)
        model.addConstr(Ei[i, k] <= Emax)

# constraint 15
for i in range(Nd_len):

    for k in range(n_evtols):
        nd = N_dash[i]
        if nd[1] == 0:
            model.addConstr(Ei[i,k] == Emax*xi[nd[0],k])

# constraint 16
for k in range(n_evtols):
    for ij in range(Ad_len):
        nd = A_dash[ij]
        if nd[1][1] > 0 and nd[1][1]  == (nd[0][1] + 1):
            j = N_dash.index(nd[1])
            i = N_dash.index(nd[0])
            if nd[0][0] != nd[1][0]:
                energy = E[ij][nd[0][0], nd[1][0]]
                model.addConstr(Ei[j, k] == ((1 - gamma) * Ei[i, k] - energy)*xij[ij, k])
            else:
                model.addConstr(g[ij, k] >= 0)
                model.addConstr(g[ij, k] <= P_max)

                # constraint 20
                i = N_dash.index(nd[0])
                model.addConstr(g[ij, k] <= (Emax - Ei[i, k]) / delta_t)
                model.addConstr(Ei[j, k] == ((1 - gamma) * Ei[i, k] + g[ij, k]*delta_t)*xij[ij, k])







# constraint 17
# for k in range(n_evtols):
#     for ij in range(Ad_len):
#         nd = A_dash[ij]
#         if nd[0][0] != nd[1][0]:
#             model.addConstr(Eij[ij, k] == E[ij][nd[0][0], nd[1][0]])
#
# for k in range(n_evtols):
#     for ij in range(Ad_len):
#         nd = A_dash[ij]
#         print(nd)
#         if nd[0][0] == nd[1][0]:
#             # constraint 18
#             # model.addConstr(g[ij,k] == -Eij[ij,k]/delta_t) # check this again
#
#             # constraint 19
#             model.addConstr(g[ij,k] >= 0)
#             model.addConstr(g[ij, k] <= P_max)
#
#             # constraint 20
#             i = N_dash.index(nd[0])
#             model.addConstr(g[ij, k] <= (Emax - Ei[i, k])/delta_t)

model.setObjective( quicksum([w[ij]*p[A_dash[ij][0][0], A_dash[ij][1][0]] if A_dash[ij][0][0] != A_dash[ij][1][0] else 0 for ij in range(Ad_len)])
                    -quicksum([g[ij,k]*xij[ij,k]*LMPR*delta_t for ij in range(Ad_len) for k in range(n_evtols)])
, GRB.MAXIMIZE)


model.setParam("TimeLimit", 3600.0)
# model.setParam('NonConvex', 1)
model.setParam('MIQCPMethod', 1)
model.setParam('InfUnbdInfo', 1)
model.setParam('DualReductions', 0)
# model.feasRelaxS(1, False, False, True)
# model.computeIIS()
# model.write("model.ilp")
model.optimize()
print("Final: ",model.status)