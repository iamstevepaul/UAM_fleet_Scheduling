"""
Author: Steve Paul 
Date: 4/21/22 """
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
n_vertiports = 5
n_evtols = 3
# set of all vertiports
N = list(range(0, n_vertiports))

# set of all arcs
A = [[i,j] for i in N for j in N]
time_len = 46
# set of time steps
T = list(range(0, time_len))
max_xy = 25
vertiport_locations = [[random.randint(0,max_xy), random.randint(0,max_xy)] for i in range(n_vertiports)]
distance_matrix = [[np.linalg.norm((np.array(p1) - np.array(p2))) for p1 in vertiport_locations] for p2 in vertiport_locations]

# set of evtols
K = list(range(0, n_evtols)) # evtols
#N_s set of all sapce time arc
A_s = [[a,t] for t in T for a in A]
q = [[[random.randint(20,200) if it1 != it2 else 0 for it1 in N] for it2 in N] for t in T] #  potential demands in each arc - this will be changed
# e = [[random.randint(20,200) if it1 != it2 else 0 for it1 in N] for it2 in N] # energe required in each ar -  this will be changed
Emax = 110 # max battery capacity of the evtols
P_max = 150
evtol_max_range = 50 # miles
evtol_energy_per_mile = Emax/evtol_max_range
c = [[random.randint(10,50) if it1 != it2 else 0 for it1 in N] for it2 in N] # cost of operating, this will not be used now, but needs to be cnhanged
C = 4 # passenger capacity of evtols
u = [6 for i in N]
p = np.array(distance_matrix)*1.5#np.array(c) + np.array(q) # fare charged to the passengers, this will be changed
d = q
E = list(np.array([distance_matrix for ti in T])*evtol_energy_per_mile)
gamma = 0.1

LMPR = 0.2



model = Model("UAM fleet scheduling")

''' Decision Variables'''
xij = model.addVars(time_len, n_vertiports, n_vertiports, n_evtols, vtype=GRB.BINARY, name="xij")
xi = model.addVars(time_len, n_vertiports, n_evtols, vtype=GRB.BINARY, name="xi")
w = model.addVars(time_len, n_vertiports, n_vertiports, vtype=GRB.INTEGER, name="w")
Cij = model.addVars(time_len, n_vertiports, n_vertiports, vtype=GRB.INTEGER, name="Cij")
Ei = model.addVars(time_len, n_vertiports, n_evtols, vtype=GRB.CONTINUOUS, name="Ei")
g = model.addVars(time_len,n_vertiports, n_vertiports, n_evtols, vtype=GRB.CONTINUOUS, name="g")

for t in T:
    for n1 in range(n_vertiports):
        for n2 in range(n_evtols):
            model.addConstr(Cij[t,n1,n2] == quicksum([C*xij[t, n1, n2, k] for k in range(n_evtols)]))

for i in T:
    for n1 in range(n_vertiports):
        for n2 in range(n_vertiports):
            model.addConstr(w[t,n1,n2] <= Cij[t,n1,n2])
            model.addConstr(w[t, n1, n2] <= d[t][n1][n2])

for k in range(n_evtols):
    model.addConstr(quicksum([xi[0,i,k] for i in range(n_vertiports)]) == 1)

for k in range(n_evtols):
    for i in range(n_vertiports):
        model.addConstr(xi[0,i,k] == quicksum([xij[0,i,j,k] for j in range(n_vertiports)]))


# constraint 11 goes here
# for k in range(n_evtols):
for t in T:
    if t != len(T) - 2:
        for h in range(n_vertiports):
            for k in range(n_evtols):
                if t > 0:
                    model.addConstr(quicksum([xij[t,i,h,k] for i in range(n_vertiports)]) == quicksum([xij[t,h,j,k] for j in range(n_vertiports)]))

# constraint 13
for i in range(n_vertiports):
    model.addConstr(quicksum([xi[0,i,k] for k in range(n_evtols)]) <= u[i])
    for t in T:
        if t > 0:
            model.addConstr(quicksum([xij[t,j,i,k] for j in range(n_vertiports) for k in range(n_evtols)]) <= u[i])

for t in T:
    for i in range(n_vertiports):
        for k in range(n_evtols):
            model.addConstr(Ei[t,i,k] >= 0)
            model.addConstr(Ei[t, i, k] <= Emax)
# constraint 15
for i in range(n_vertiports):
    for k in range(n_evtols):
        model.addConstr(Ei[0,i,k] == xi[0,i,k]*Emax)

# constraint 16 goes here

for t in T:
    for i in range(n_vertiports):
        for j in range(n_vertiports):
            for k in range(n_evtols):
                s= t+1
                if s != len(T):
                    model.addConstr(Ei[s,j,k] == (Ei[t,i,k] - E[t][i][j])*xij[t,i,j,k])

# cosntraint 17 is nt required


#c constraint 18
for t in T:
    for i in range(n_vertiports):
        for j in range(n_vertiports):
            for k in range(n_evtols):
                if i==j:
                    model.addConstr(g[t,i,j,k] == -2*E[t][i][j])# .5 indicates the time step length of 30 minutes or .5 hours
                    model.addConstr(g[t,i,j,k] >= 0)
                    model.addConstr(g[t, i, j, k] <= P_max)
                    model.addConstr(g[t, i, j, k]*.5 <= Emax - Ei[t,i,k])

# setting the onjective function
RT = quicksum([p[i,j]*w[t,i,j] if i!=j else 0 for j in range(n_vertiports) for i in range(n_vertiports) for t in T])
# CO = quicksum([c[i][j]*xij[t,i,j,k] if i!=j else 0 for k in range(n_evtols) for j in range(n_vertiports) for i in range(n_vertiports) for t in T])
CC = quicksum([g[t,i,j,k]*xij[t,i,j,k]*LMPR if i!=j else 0 for k in range(n_evtols) for j in range(n_vertiports) for i in range(n_vertiports) for t in T])
model.setObjective(quicksum([RT, -CC]), GRB.MAXIMIZE)

# model.setParam('MIPFocus', 3)
model.setParam("TimeLimit", 3600.0)
# model.setParam('NonConvex', 1)
model.setParam('MIQCPMethod', 0)
model.setParam('InfUnbdInfo', 1)
model.setParam('DualReductions', 0)


# if model.status == GRB.INFEASIBLE:
#     var = model.getVars()
#     var_o = var[m * h * (n + 1) ** 2 + m * h * (n + 1) * 2:]
#     ubpen = [7200] * (m * h * (n + 1))
#     model.feasRelax(1, False, var_o, None, ubpen, None, None)
#     model.optimize()
# if model.status == GRB.INFEASIBLE:
model.feasRelaxS(1, False, False, True)
model.optimize()

print("Final: ",model.status)


