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

# Number of tasks
n = 12  # Even integer only

# Starting point location index
O = list(range(1, int(n / 2) + 1))

# Ending point location index
D = list(range(int(n / 2) + 1, n + 1))

# Number of Robots
m = 3

# Max number of tour for a robot
h = 2  # int(n/m+1)

# Max capacity for each robot
C = 4

# Speed of robot
speed = 2.778

# Deadline of tasks
deadline = np.zeros(n)
deadline[0:int(n / 2)] = np.array([(random.randint(510, 1800)) for i in range(int(n / 2))])  # 360 1800
deadline[int(n / 2):n] = np.array([(random.randint(2200, 3600)) for i in range(int(n / 2))])  # 2000 3600

# Number of robots needed for each task
q = [random.randint(1, 3) for i in range(int(n / 2))]
e = []
for i in range(len(q)):
    q.append(-q[i])
    if q[i] <= 2:
        e.append(1)
    else:
        e.append(2)

for i in range(len(e)):
    e.append(e[i])
# e = [1,2,1,2,1,2,1,2]
# q = [1,3,2,3,-1,-3,-2,-3]
E = [i for i, x in enumerate(e) if x >= 2]
for i in range(len(E)):
    E[i] = E[i] + 1

SE = [i for i, x in enumerate(e) if x < 2]
for i in range(len(SE)):
    SE[i] = SE[i] + 1

# Max distance travelled per tour for each robot
del_range = 4000

# Get random locations
n = n + 1
X = [(random.randint(0, 1000)) for i in range(n)]
X.append(X[0])
Y = [(random.randint(0, 1000)) for i in range(n)]
Y.append(Y[0])
# depot = [random.randint(0,100),random.randint(0,100)]
location = np.column_stack((X, Y))

depot, tasks = location[0, :], location[1:, :]

# Get Euclidean distance and decision variables
dist = np.empty((n + 1, n + 1))
for i in range(n + 1):
    for j in range(n + 1):
        dist[i, j] = np.sqrt((X[i] - X[j]) ** 2 + (Y[i] - Y[j]) ** 2)
        if i == j:
            dist[i, j] = 100 * 100
        continue
dist[(0, 0)] = 100 * 100
dist[(0, -1)] = 100 * 100
dist[(-1, 0)] = 100 * 100
dist[(-1, -1)] = 100 * 100

# def MILP_CollectiveTransport(n,m,h,C,speed,e,q,O,D,dist,tasks,depot):


# Create Optimization model
model = Model("Collective_Transport")

''' Decision Variables'''
x = model.addVars(m, h, n + 1, n + 1, vtype=GRB.BINARY, name="x")
y = model.addVars(m, h, n + 1, vtype=GRB.INTEGER, name="y")
z = model.addVars(m, h, n + 1, vtype=GRB.INTEGER, name="z")
# u = model.addVars(m,h,n+1,vtype=GRB.INTEGER,lb=2,ub=n,name="u")
o = model.addVars(m, h, n + 1, max(e), vtype=GRB.CONTINUOUS, name="o")

# Set variables x to zero in specific condition
for r in range(m):
    for s in range(h):
        z[r, s, 0] = 0
        z[r, s, n] = 0

        for i in range(n + 1):
            x[r, s, n, i] = 0
            x[r, s, i, 0] = 0
            x[r, s, i, i] = 0
            for ii in range(max(e)):
                model.addConstr(o[r, s, i, ii] >= 0)

'''Constraints'''
for r in range(m):
    for s in range(h):
        if s == 0:
            # C7 Starting point connectivity constraint
            model.addConstr(quicksum(x[r, s, 0, j] for j in O) == 1)

            # C8 Ending point connectivity constraint
            model.addConstr(quicksum(x[r, s, i, n] for i in D) == 1)

        else:
            # C10 implicit constraint from constraint 8
            model.addConstr((y[r, s, 0] == 1) >> (quicksum(x[r, s, i, n] for i in D) == 1))
            # C9 implicit constraint from constraint 7
            model.addConstr((y[r, s, 0] == 1) >> (quicksum(x[r, s, 0, i] for i in O) == 1))

        model.addConstr(y[r, s, 0] <= 1)
        model.addConstr(y[r, s, n] <= 1)

        # C14 Range Constraint
        model.addConstr(quicksum(quicksum(dist[(i, j)] * x[r, s, i, j] for j in range(n + 1))
                                 for i in range(n + 1)) <= del_range)

        for i in range(n):
            # C4 Relation constraint for varaible x and y
            model.addConstr(quicksum(x[r, s, i, j + 1] for j in range(n)) == y[r, s, i])  # y[r,s,i]

        for a in O:
            # C5 Starting and ending point connectivity constraint
            model.addConstr(quicksum(x[r, s, a, j + 1] for j in range(n - 1)) -
                            quicksum(x[r, s, j + 1, a + int(n / 2)] for j in range(n - 1)) >= 0)

            # C17 Capacity Constraint
            model.addConstr(z[r, s, a] >= q[a])
            model.addConstr(z[r, s, a] <= C)

            # C18 Capacity Constraint
            model.addConstr(z[r, s, a + int(n / 2)] >= 0)
            model.addConstr(z[r, s, a + int(n / 2)] <= C - q[a])

            for ii in range(max(e)):
                # precedent constraint
                model.addConstr(o[r, s, a, ii] + dist[(a, a + int(n / 2))] / speed <= o[r, s, a + int(n / 2), ii])

        for k in range(n - 1):
            # C6 Connectivity constraint
            model.addConstr(quicksum(x[r, s, i, k + 1] for i in range(n)) -
                            quicksum(x[r, s, k + 1, j + 1] for j in range(n)) == 0)

            for zz in range(max(e)):
                # Deadline constraint
                model.addConstr(o[r, s, k + 1, zz] <= deadline[k])

            for j in range(n - 1):

                # C16 Deadline Constraint
                model.addConstr(x[r, s, k + 1, j + 1] * (z[r, s, k + 1] + q[j] - z[r, s, j + 1]) == 0)

                for zzz in range(max(e)):
                    # precedent constraint
                    model.addConstr(o[r, s, j + 1, zzz] >= (o[r, s, k + 1, zzz] + dist[(k + 1, j + 1)] / speed) * x[
                        r, s, k + 1, j + 1])

for i in E:
    # C11 Constraint for robots to perform tasks that require two or more robots
    model.addConstr(quicksum(quicksum(y[r, s, i] for s in range(h)) for r in range(m)) == e[i - 1])

for i in SE:
    # Constraint for robots to perform tasks that require single robot
    model.addConstr(quicksum(quicksum(y[r, s, i] for s in range(h)) for r in range(m)) == e[i - 1])

'''Objective Function'''
# Create Objective Function (Total distance travelled)
model.setObjective(quicksum
                   (quicksum
                    (quicksum
                     (quicksum(x[r, s, i, j] * dist[(i, j)]
                               for j in range(n))
                      for i in range(n))
                     for s in range(h))
                    for r in range(m)), GRB.MINIMIZE)

' Check Gurobi website to modify the appropriate parameters '
# model.tune()
# model.setParam('Method', 2)
model.setParam('MIPFocus', 3)
model.setParam("TimeLimit", 3600.0)
model.setParam('NonConvex', 2)
# model.setParam("Cuts", 2)
'''Solve the problem'''

if model.status == GRB.INFEASIBLE:
    var = model.getVars()
    var_o = var[m * h * (n + 1) ** 2 + m * h * (n + 1) * 2:]
    ubpen = [7200] * (m * h * (n + 1))
    model.feasRelax(1, False, var_o, None, ubpen, None, None)
    model.optimize()

model.optimize()

print('Computation Time %s seconds' % (time.time() - start))

'------------------------------------------------------------------------'
'''Solution Plotting'''


def plot_tours(solution_x):
    tours = [[i, j] for i in range(solution_x.shape[0]) for j in range(solution_x.shape[1]) if solution_x[i, j] == 1]
    for tour in tours:
        plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="black", linewidth=0.5)

    array_tours = np.array(tours)
    route = [0]
    next_index = 0
    for i in range(len(tours)):
        next_point = int(array_tours[next_index, 1])
        next_index = np.argwhere(array_tours[:, 0] == next_point)
        route.append(next_point)

    tour_x = []
    tour_y = []
    for i in route:
        tour_x.append(X[i])
        tour_y.append(Y[i])

    u = np.diff(tour_x)
    v = np.diff(tour_y)
    pos_x = tour_x[:-1] + u / 2
    pos_y = tour_y[:-1] + v / 2
    norm = np.sqrt(u ** 2 + v ** 2)

    colors = ['r', 'orange', 'g', 'b', 'c', 'm', 'y', 'k', 'blueviolet', 'lawngreen']  # ,'m', 'y', 'k']
    for ii in range(len(O)):
        if e[ii] > 1:
            plt.scatter(X[O[ii]], Y[O[ii]], marker='^', color=colors[ii], s=70, label='Staring Points')
            plt.scatter(X[D[ii]], Y[D[ii]], marker=',', color=colors[ii], s=70, label='Ending Points')
        else:
            plt.scatter(X[O[ii]], Y[O[ii]], marker='^', color=colors[ii], s=10, label='Staring Points')
            plt.scatter(X[D[ii]], Y[D[ii]], marker=',', color=colors[ii], s=10, label='Ending Points')
    plt.scatter(depot[0], depot[1], s=80, marker='*', label='Depot')
    plt.xlabel("X"), plt.ylabel("Y"), plt.title("Tours"), plt.legend(bbox_to_anchor=(1.05, 1))
    plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")
    plt.show()


def plot_tours3(X_sol11, X_sol21, X_sol31):
    # ,X_sol12,X_sol22,X_sol32
    tours1 = [[i, j] for i in range(X_sol11.shape[0]) for j in range(X_sol11.shape[1]) if X_sol11[i, j] == 1]
    tours2 = [[i, j] for i in range(X_sol21.shape[0]) for j in range(X_sol21.shape[1]) if X_sol21[i, j] == 1]
    tours3 = [[i, j] for i in range(X_sol31.shape[0]) for j in range(X_sol31.shape[1]) if X_sol31[i, j] == 1]

    for tour in tours1:
        if tour == tours1[-1]:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="red", alpha=0.5, linewidth=1,
                     label='Robot1')
        else:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="red", alpha=0.5, linewidth=1)

    for tour in tours2:
        if tour == tours2[-1]:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="lime", alpha=0.6, linewidth=3,
                     label='Robot2')
        else:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="lime", alpha=0.6, linewidth=3)

    for tour in tours3:
        if tour == tours3[-1]:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="navy", alpha=0.3, linewidth=5,
                     label='Robot3')
        else:
            plt.plot([X[tour[0]], X[tour[1]]], [Y[tour[0]], Y[tour[1]]], color="navy", alpha=0.3, linewidth=5)
    plt.legend(bbox_to_anchor=(1.05, 1))

    # for tour in tours4:
    #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "red", alpha=0.15,linewidth=0.5)

    # for tour in tours5:
    #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "green", alpha=0.15,linewidth=0.5)

    # for tour in tours6:
    #     plt.plot([X[tour[0]],X[tour[1]]], [Y[tour[0]],Y[tour[1]]], color = "blue", alpha=0.15,linewidth=0.5)

    def arrows(tours):
        array_tours = np.array(tours)
        route = [0]
        next_index = 0
        for i in range(len(tours)):
            next_point = int(array_tours[next_index, 1])
            next_index = np.argwhere(array_tours[:, 0] == next_point)
            route.append(next_point)

        tour_x = []
        tour_y = []
        for i in route:
            tour_x.append(X[i])
            tour_y.append(Y[i])

        u = np.diff(tour_x)
        v = np.diff(tour_y)
        pos_x = tour_x[:-1] + u / 2
        pos_y = tour_y[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        plt.quiver(pos_x, pos_y, u / norm, v / norm, angles="xy", zorder=5, pivot="mid")

    colors = ['r', 'orange', 'g', 'b', 'c', 'm', 'y', 'k', 'blueviolet', 'lawngreen']  # ,'m', 'y', 'k']
    for ii in range(len(O)):
        if e[ii] > 1:
            plt.scatter(X[O[ii]], Y[O[ii]], marker='^', color=colors[ii], s=70, label='Pickup Task')
            plt.scatter(X[D[ii]], Y[D[ii]], marker=',', color=colors[ii], s=70, label='Delivery Task')
        else:
            plt.scatter(X[O[ii]], Y[O[ii]], marker='^', color=colors[ii], s=25, label='Pickup Task')
            plt.scatter(X[D[ii]], Y[D[ii]], marker=',', color=colors[ii], s=25, label='Delivery Task')
    plt.scatter(depot[0], depot[1], s=100, marker='*', label='Depot')
    plt.xlabel("X"), plt.ylabel("Y"), plt.title("MINLP (Total Traveling Distance = 6504.94m)"), plt.legend(
        bbox_to_anchor=(1.05, 1))
    arrows(tours1)
    arrows(tours2)
    arrows(tours3)
    plt.show()


solx = model.getAttr('x')
solx = solx[0:m * h * (n + 1) ** 2]

# solx = [[[] for i in range(m)] for ii in range(h)]


"h=2"
solx11 = solx[0:(n + 1) ** 2]
solx12 = solx[(n + 1) ** 2:2 * (n + 1) ** 2]
solx21 = solx[2 * (n + 1) ** 2:3 * (n + 1) ** 2]
solx22 = solx[3 * (n + 1) ** 2:4 * (n + 1) ** 2]
solx31 = solx[4 * (n + 1) ** 2:5 * (n + 1) ** 2]
solx32 = solx[5 * (n + 1) ** 2:6 * (n + 1) ** 2]

X_sol11 = np.empty([n + 1, n + 1])
X_sol12 = np.empty([n + 1, n + 1])
X_sol21 = np.empty([n + 1, n + 1])
X_sol22 = np.empty([n + 1, n + 1])
X_sol31 = np.empty([n + 1, n + 1])
X_sol32 = np.empty([n + 1, n + 1])

index = 0
for i in range(n + 1):
    for j in range(n + 1):
        X_sol11[i, j] = round(solx11[index])
        X_sol12[i, j] = round(solx12[index])
        X_sol21[i, j] = round(solx21[index])
        X_sol22[i, j] = round(solx22[index])
        X_sol31[i, j] = round(solx31[index])
        X_sol32[i, j] = round(solx32[index])
        index = index + 1

plot_tours(X_sol11)
plot_tours(X_sol21)
plot_tours(X_sol31)
plot_tours(X_sol12)
plot_tours(X_sol22)
plot_tours(X_sol32)

plot_tours3(X_sol11, X_sol21, X_sol31)
plot_tours3(X_sol12, X_sol22, X_sol32)

"h=3"
# solx11 = solx[0:(n+1)**2]
# solx12 = solx[(n+1)**2:2*(n+1)**2]
# solx13 = solx[2*(n+1)**2:3*(n+1)**2]
# solx21 = solx[3*(n+1)**2:4*(n+1)**2]
# solx22 = solx[4*(n+1)**2:5*(n+1)**2]
# solx23 = solx[5*(n+1)**2:6*(n+1)**2]
# solx31 = solx[6*(n+1)**2:7*(n+1)**2]
# solx32 = solx[7*(n+1)**2:8*(n+1)**2]
# solx33 = solx[8*(n+1)**2:9*(n+1)**2]


# X_sol11 = np.empty([n+1, n+1])
# X_sol12 = np.empty([n+1, n+1])
# X_sol13 = np.empty([n+1, n+1])
# X_sol21 = np.empty([n+1, n+1])
# X_sol22 = np.empty([n+1, n+1])
# X_sol23 = np.empty([n+1, n+1])
# X_sol31 = np.empty([n+1, n+1])
# X_sol32 = np.empty([n+1, n+1])
# X_sol33 = np.empty([n+1, n+1])

# index = 0
# for i in range(n+1):
#     for j in range(n+1):
#         X_sol11[i, j] = round(solx11[index])
#         X_sol12[i, j] = round(solx12[index])
#         X_sol13[i, j] = round(solx13[index])
#         X_sol21[i, j] = round(solx21[index])
#         X_sol22[i, j] = round(solx22[index])
#         X_sol23[i, j] = round(solx23[index])
#         X_sol31[i, j] = round(solx31[index])
#         X_sol32[i, j] = round(solx32[index])
#         X_sol33[i, j] = round(solx33[index])
#         index = index+1


# plot_tours(X_sol11)
# plot_tours(X_sol21)
# plot_tours(X_sol31)
# plot_tours(X_sol12)
# plot_tours(X_sol22)
# plot_tours(X_sol32)
# plot_tours(X_sol13)
# plot_tours(X_sol23)
# plot_tours(X_sol33)


# plot_tours3(X_sol11,X_sol21,X_sol31)
# plot_tours3(X_sol12,X_sol22,X_sol32)
# plot_tours3(X_sol13,X_sol23,X_sol33)