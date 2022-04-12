"""
Author: Steve Paul 
Date: 3/9/22 """

import gym
import torch as th
import  numpy as np
from eVTOL import eVTOL
from Vertiport import Vertiport
from gym.spaces import Discrete, MultiBinary, Box, Dict
import networkx as nx
from topology import *
import scipy.sparse as sp
from persim import wasserstein, bottleneck
from collections import defaultdict

class UAMEnv(gym.Env):

    def __init__(self,
                 vertiport_locations,
                 evtols_initial_locations,
                 n_vertiports,
                 n_evtols,
                 demand_model = None,
                 pricing_model = None,
                 time_horizon = None,
                 passenger_pricing_model = None,
                 take_off_time = 0.25,
                 landing_time = 0.25,
                 vertiports_n_parked = None,
                 vertiports_max_parked = None,
                 vertiports_max_charged = None
                 ):
        self.n_vertiports = n_vertiports
        self.n_evtols = n_evtols
        self.action_space = Discrete(n_vertiports)
        while ((evtols_initial_locations == evtols_initial_locations.T.mode().values).to(th.int64).sum()).item() > 3:
            evtols_initial_locations = self.generate_evtols_starting_locations()


        self.vertiports = [
            Vertiport(
                id=i,
                location=vertiport_locations[i,:],
                max_evotls_park = 6,
                max_evtol_charge=6
            )
            for i in range(n_vertiports)]

        self.take_off_time = take_off_time # 15 minutes or .25 hours
        self.landing_time = landing_time # 15 minutes or .25 hours
        self.evtols = [
            eVTOL(
                id=i, max_passenger=6,
                  location=evtols_initial_locations[i,:],
                  take_off_time = take_off_time,
                landing_time=landing_time
            )
            for i in range(n_evtols)]
        self.vertiport_locations = vertiport_locations
        self.demand_model = demand_model
        self.electricity_pricing_model = self.generate_electricity_pricing_model()
        self.time_horizon = time_horizon
        self.vertiports_distance_matrix = th.cdist(vertiport_locations, vertiport_locations, p=2)
        self.passenger_pricing_model = self.generate_passenger_pricing_model()
        self.current_time = th.tensor([6.00]) # read as 6 am
        self.evtols_next_decision_time = 6.00+th.zeros((n_evtols, 1))
        self.evtol_taking_decision = 0 # integer keeps track of the id of the evtol taking decision
        self.total_electricity_charge = 0.0
        self.total_ticket_collection = 0.0
        self.i = 0
        self.evtols_locations = th.zeros((n_evtols,1))
        for i in range(n_evtols):
            self.evtols_locations[i,0] = self.evtols[i].current_location
        self.vertiports_n_charged = th.zeros((n_vertiports,1))
        if vertiports_n_parked:
            self.vertiports_n_parked = vertiports_n_parked
        else:
            self.vertiports_n_parked = th.zeros((n_vertiports, 1))
            for i in range(n_evtols):
                self.vertiports_n_parked[self.evtols[i].current_location, 0] += 1
                self.vertiports[self.evtols[i].current_location].update_parked_evtols(1)

        if vertiports_max_parked:
            self.vertiports_max_parked = vertiports_max_parked
        else:
            self.vertiports_max_parked = th.zeros((n_vertiports, 1))
            for i in range(n_vertiports):
                self.vertiports_max_parked[i,0] = self.vertiports[i].max_evotls_park

        if vertiports_max_charged:
            self.vertiports_max_charged = vertiports_max_charged
        else:
            self.vertiports_max_charged = th.zeros((n_vertiports, 1))
            for i in range(n_vertiports):
                self.vertiports_max_charged[i, 0] = self.vertiports[i].max_evtol_charge
        self.time_varying_demand_model = self.get_time_varying_demand()

        vertiport_graph = self.get_vertiport_graph()
        evtol_graph = self.get_evtol_graph()
        self.observation_space = Dict(
            dict(
                vertiport_location=Box(low=0, high=60, shape=vertiport_locations.shape),
                mask=Box(low=0, high=1, shape=(self.n_vertiports,1)),
                demand=Box(low=0, high=100, shape=demand_model.shape),
                evtols_locations=Box(low=0, high=n_evtols, shape=self.evtols_locations.shape),
                electricity_pricing_model=Box(low=0, high=n_evtols, shape=self.electricity_pricing_model.shape),
                vertiports_distance_matrix=Box(low=0, high=100, shape=self.vertiports_distance_matrix.shape),
                evtols_next_decision_time=Box(low=6.00, high=18.00, shape=self.evtols_next_decision_time.shape),
                evtol_taking_decision=Discrete(n_evtols),
                evtol_taking_decision_location=Box(low=0, high=100, shape=(1,2)),
                vertiports_n_parked=Box(low=0, high=3, shape=self.vertiports_n_parked.shape),
                vertiports_n_charged=Box(low=0, high=3, shape=self.vertiports_n_charged.shape),
                vertiport_graph_nodes=Box(low=0, high=1, shape=vertiport_graph["nodes"].shape),
                vertiport_graph_adjacency= Box(low=0, high=1, shape=vertiport_graph["adjacency"].shape),
                evtol_graph_nodes=Box(low=0, high=1, shape=evtol_graph["nodes"].shape),
                evtol_graph_adjacency= Box(low=0, high=1, shape=evtol_graph["adjacency"].shape),
            )
        )


    def compute_reward(self):
        reward = .1
        return reward

    def update_electricity_charge(self):
        pass

    def step(self, action):
        # action space: [no action, locations]
        # action taken should be in such a way that UAM taking decision can only go to vertiports where there is a vacancy to park or charge
        evtol_taking_decision_id = self.evtol_taking_decision
        current_time = self.current_time.clone()

        evtol_taking_decision = self.evtols[evtol_taking_decision_id]

        # update evtols needs decision

        info = {}
        reward = 0.0
        done = False

        self.i = self.i+1
        # print(self.current_time)
        if action == evtol_taking_decision.current_location:
            # update the status of the evtol taking decision as 0

            if evtol_taking_decision.status == 1:
                self.vertiports[evtol_taking_decision.current_location].update_charging_evtols(-1)
                self.vertiports[evtol_taking_decision.current_location].update_parked_evtols(1)

                self.vertiports_n_charged[evtol_taking_decision.current_location] -= 1
                if self.vertiports_n_charged[evtol_taking_decision.current_location] < 0:
                    raise ValueError
                if self.vertiports_n_charged[evtol_taking_decision.current_location] > self.vertiports[evtol_taking_decision.current_location].max_evtol_charge:
                    raise ValueError
                if self.vertiports_n_charged[evtol_taking_decision.current_location] != self.vertiports[evtol_taking_decision.current_location].n_evtols_charging:
                    raise ValueError

                self.vertiports_n_parked[evtol_taking_decision.current_location] += 1
                if self.vertiports_n_parked[evtol_taking_decision.current_location] < 0:
                    raise ValueError
                if self.vertiports_n_parked[evtol_taking_decision.current_location] > self.vertiports[evtol_taking_decision.current_location].max_evotls_park:
                    raise ValueError

                if self.vertiports_n_parked[evtol_taking_decision.current_location] != self.vertiports[evtol_taking_decision.current_location].n_evtol_parked:
                    raise ValueError

            # elif evtol_taking_decision.status == 0:
            #     self.vertiports[evtol_taking_decision.current_location].update_parked_evtols(-1)
            self.update_evtols_status(self.evtol_taking_decision, 0)


            evtol_next_decision_time = current_time + 0.25


            # print(self.current_time," old     111111111")
            self.update_evtol_decision_time(self.evtol_taking_decision, evtol_next_decision_time)
            self.update_evtols_decision_time(self.evtol_taking_decision, evtol_next_decision_time)
            # print(self.current_time, "      22222222")
            self.current_time = current_time

        else:
            # update the parking information of the vertiport

            # update the charging number of the vertiport
            self.vertiports[action].update_charging_evtols(1)
            self.vertiports_n_charged[action] += 1

            if evtol_taking_decision.status == 1:
                self.vertiports[evtol_taking_decision.current_location].update_charging_evtols(-1)
                self.vertiports_n_charged[evtol_taking_decision.current_location] -= 1
            elif evtol_taking_decision.status == 0:
                self.vertiports[evtol_taking_decision.current_location].update_parked_evtols(-1)
                self.vertiports_n_parked[evtol_taking_decision.current_location] -= 1



            # update the status of the evtol taking decision as 2
            self.update_evtols_status(self.evtol_taking_decision, 2)

            # update the takeoff time and landing time of the evtol taking decision
            self.update_evtol_next_flight_time(self.evtol_taking_decision, current_time)
            evtol_current_location_id = evtol_taking_decision.current_location
            evtol_speed = evtol_taking_decision.speed
            flight_time = self.vertiports_distance_matrix[evtol_current_location_id, action]/evtol_speed
            discharge = flight_time*0.9 # this will be changed
            charge_time = discharge # this will be changed
            self.evtols[evtol_taking_decision_id].battery.current_battery_charge -= discharge

            #####
            electricity_price = discharge*self.electricity_pricing_model
            self.total_electricity_charge = self.total_electricity_charge + electricity_price

            take_off_time = evtol_taking_decision.take_off_time
            landing_time = evtol_taking_decision.landing_time
            evtol_next_decision_time = current_time + take_off_time + flight_time + landing_time + charge_time
            # print(self.current_time, " new     111111111")
            self.update_evtol_decision_time(self.evtol_taking_decision, evtol_next_decision_time)
            # print(self.current_time, "      2222222222")
            self.update_evtols_decision_time(self.evtol_taking_decision, evtol_next_decision_time)
            # print(self.current_time, "      333333333")
            self.current_time = current_time


            # reduce the charge now

            # update the remaining demand depending on the number of passengers that will be transported by the evtol taking decision
            n_passengers = self.update_demand(evtol_taking_decision.current_location, action)
            ticket_collection = self.passenger_pricing_model[evtol_taking_decision.current_location, action]*n_passengers
            self.total_ticket_collection = self.total_ticket_collection + ticket_collection

            # update the current location and the next location of the evtl taking decision
            self.update_evtols_locations(self.evtol_taking_decision, action)

            # compute the cost
                # cost of charging

            # compute ticket sale

        self.evtols[evtol_taking_decision_id].update_location(action)

        # find the next evtols which will be taking decision
            # one of the idle ones and one which can have atleast one destination
            # an evtol which has just taken a decision will be masked
            # decrease the charge of the evtol
            # if this evtol was in motion, set its status as charging
        # find the next decision time
        # update the current time
        # update previous update time

        next_dec_time, next_evtol = self.find_next_update_time()
        # if next_dec_time > self.time_horizon:
        #     dt = 0
        self.current_time = next_dec_time
        self.evtol_taking_decision = next_evtol
        new_evtol_taking_decision = self.evtols[self.evtol_taking_decision]
        self.evtols[evtol_taking_decision_id].battery.current_battery_charge = self.evtols[evtol_taking_decision_id].battery.battery_max
        if new_evtol_taking_decision.status == 2: # if the evtol was travelling, set its status to 1
            self.evtols[self.evtol_taking_decision].status = 1

            # full charge now

        if self.current_time > self.time_horizon:
            done = True
            # reward = ((self.total_ticket_collection - self.total_electricity_charge)).item()
            reward = ((self.total_ticket_collection - self.total_electricity_charge)/(self.time_varying_demand_model["demand"]*self.passenger_pricing_model).sum()).item()
            info = {"is_success": done,
                    "episode": {
                        "r": reward,
                        "l": self.i
                        }
                    }

        new_state = self.get_new_state()
        return new_state, reward, done, info

    def get_mask(self):
        evtol_id = self.evtol_taking_decision
        mask = th.ones((self.n_vertiports, 1))
        max_charged_locs = (self.vertiports_n_charged == self.vertiports_max_charged).nonzero()
        evtol_status = self.evtols[evtol_id].status
        evtol_current_location = self.evtols[evtol_id].current_location
        if evtol_status == 1:
            if self.vertiports_n_parked[evtol_current_location, 0] ==  self.vertiports_max_parked[evtol_current_location, 0]:
                mask[evtol_current_location, 0] = 0
        if max_charged_locs.shape[0] != 0:
            max_charged_locs = max_charged_locs[:,0]
            mask[max_charged_locs, 0] = 0

        return mask

    def get_new_state(self):
        vertiport_graph = self.get_vertiport_graph()
        evtol_graph = self.get_evtol_graph()
        return {
                "vertiport_location":self.vertiport_locations,
                "mask": self.get_mask(),
                "demand":self.demand_model,
                "evtols_locations": self.evtols_locations,
                "electricity_pricing_model": self.electricity_pricing_model,
                "vertiports_distance_matrix": self.vertiports_distance_matrix,
                "evtol_taking_decision":self.evtol_taking_decision,
                "evtols_next_decision_time": self.evtols_next_decision_time,
                "evtol_taking_decision_location":self.vertiport_locations[self.evtols[self.evtol_taking_decision].current_location, :],
                "vertiports_n_parked":self.vertiports_n_parked,
                "vertiports_n_charged": self.vertiports_n_charged,
                "vertiport_graph_nodes": vertiport_graph["nodes"],
                "vertiport_graph_adjacency": vertiport_graph["adjacency"],
                "evtol_graph_nodes": evtol_graph["nodes"],
                "evtol_graph_adjacency": evtol_graph["adjacency"],
                "time_varying_demand": self.time_varying_demand_model,
                "passenger_pricing": self.passenger_pricing_model,
        }

    def update_evtols_locations(self, evtol_id, location):
        self.evtols[evtol_id].current_location = location

    def update_evtol_decision_time(self, evtol_id, decision_time):
        self.evtols[evtol_id].next_decision_time = decision_time

    def update_evtol_next_flight_time(self, evtol_id, flight_time):
        self.evtols[evtol_id].next_flight_time = flight_time

    def update_evtols_status(self, evtol_id, status):
        self.evtols[evtol_id].status = status

    def update_evtol_taking_decision(self, evto_id):
        self.evtol_taking_decision = evto_id

    def update_evtols_decision_time(self, evtol_id, time):
        self.evtols_next_decision_time[evtol_id,0] = time



    def find_next_update_time(self):
        ind = th.argmin(self.evtols_next_decision_time[:,0])
        val = self.evtols_next_decision_time[ind,0]
        # print(ind, val)
        return val, ind

    def update_demand(self, source, destination):

        n_passenger = min(self.evtols[self.evtol_taking_decision].max_passenger,self.demand_model[source, destination])
        self.evtols[self.evtol_taking_decision].update_current_passengers(n_passenger)
        self.demand_model[source, destination] = self.demand_model[source, destination] - n_passenger
        time_id = (self.time_varying_demand_model["time_points"] >= self.current_time).nonzero()[0][0]
        self.time_varying_demand_model["demand"][time_id, source, destination] -= n_passenger
        return n_passenger

    def reset(self):
        evtols_initial_locations = self.generate_evtols_starting_locations()
        while ((evtols_initial_locations == evtols_initial_locations.T.mode().values).to(th.int64).sum()).item() > 3:
            evtols_initial_locations = self.generate_evtols_starting_locations()
        self.evtols = [
            eVTOL(
                id=i, max_passenger=6,
                location=evtols_initial_locations[i, :],
                take_off_time=self.take_off_time,
                landing_time=self.landing_time
            )
            for i in range(self.n_evtols)]
        self.demand_model = self.generate_demand_model()
        self.passenger_pricing_model = self.generate_passenger_pricing_model()
        self.current_time = th.tensor([6.00]) # read as 6 am
        self.evtols_next_decision_time = 6.00 + th.zeros((self.n_evtols, 1))
        self.evtol_taking_decision = 0  # integer keeps track of the id of the evtol taking decision
        self.total_electricity_charge = 0.0
        self.total_ticket_collection = 0.0
        self.i = 0
        self.evtols_locations = self.generate_evtols_starting_locations()
        self.electricity_pricing_model = self.generate_electricity_pricing_model()
        self.vertiports_n_charged = th.zeros((self.n_vertiports, 1))

        self.vertiports_n_parked = th.zeros((self.n_vertiports, 1))
        for i in range(self.n_vertiports):
            self.vertiports[i].n_evtol_parked = 0
            self.vertiports[i].n_evtols_charging = 0
        for i in range(self.n_evtols):
            self.vertiports_n_parked[self.evtols[i].current_location, 0] += 1
            self.vertiports[self.evtols[i].current_location].update_parked_evtols(1)
        self.time_varying_demand_model = self.get_time_varying_demand()

        # if vertiports_max_parked:
        #     self.vertiports_max_parked = vertiports_max_parked
        # else:
        #     self.vertiports_max_parked = th.zeros((n_vertiports, 1))
        #     for i in range(n_vertiports):
        #         self.vertiports_max_parked[i, 0] = self.vertiports[i].max_evotls_park

        # if vertiports_max_charged:
        #     self.vertiports_max_charged = vertiports_max_charged
        # else:
        #     self.vertiports_max_charged = th.zeros((n_vertiports, 1))
        #     for i in range(n_vertiports):
        #         self.vertiports_max_charged[i, 0] = self.vertiports[i].max_evtol_charge

        return self.get_new_state()

    def var_preprocess(self, adj, r):
        adj_ = adj + sp.eye(adj.shape[0])
        adj_ = adj_ ** r
        adj_[adj_ > 1] = 1
        rowsum = adj_.sum(1).A1
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5))
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).T.dot(degree_mat_inv_sqrt).tocsr()
        return adj_normalized

    def get_topo_laplacian(self, data):

        X_loc = data['vertiport_graph_nodes'][None, :, :]
        # distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)[0]
        distance_matrix = th.cdist(th.tensor(X_loc), th.tensor(X_loc), p=2)[0]

        adj_ = np.float32(distance_matrix < 0.8)

        dt = defaultdict(list)
        for i in range(adj_.shape[0]):
            n_i = adj_[i, :].nonzero()[0].tolist()

            dt[i] = n_i

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(dt))
        adj_array = adj.toarray().astype(np.float32)
        var_laplacian = self.var_preprocess(adj=adj, r=2).toarray()

        secondorder_subgraph = k_th_order_weighted_subgraph(adj_mat=adj_array, w_adj_mat=distance_matrix, k=2)

        reg_dgms = list()
        for i in range(len(secondorder_subgraph)):
            # print(i)
            tmp_reg_dgms = simplicial_complex_dgm(secondorder_subgraph[i])
            if tmp_reg_dgms.size == 0:
                reg_dgms.append(np.array([]))
            else:
                reg_dgms.append(np.unique(tmp_reg_dgms, axis=0))

        reg_dgms = np.array(reg_dgms)

        row_labels = np.where(var_laplacian > 0.)[0]
        col_labels = np.where(var_laplacian > 0.)[1]

        topo_laplacian_k_2 = np.zeros(var_laplacian.shape, dtype=np.float32)

        for i in range(row_labels.shape[0]):
            tmp_row_label = row_labels[i]
            tmp_col_label = col_labels[i]
            tmp_wasserstin_dis = wasserstein(reg_dgms[tmp_row_label], reg_dgms[tmp_col_label])
            # if tmp_wasserstin_dis == 0.:
            #     topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / 1e-1
            #     topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / 1e-1
            # else:
            topo_laplacian_k_2[tmp_row_label, tmp_col_label] = 1. / (tmp_wasserstin_dis + 1)
            topo_laplacian_k_2[tmp_col_label, tmp_row_label] = 1. / (tmp_wasserstin_dis + 1)

        return topo_laplacian_k_2


    def render(self, mode="human"):
        pass

    def get_vertiport_graph(self):
        # node properties:
            # x,y location
        Vertiport_graph_nods = []
        for vertiport in self.vertiports:
            props = [vertiport.location[0].item()/60,
                     vertiport.location[1].item()/60,
                     vertiport.n_evtol_parked/vertiport.max_evotls_park,
                     vertiport.n_evtols_charging/vertiport.max_evtol_charge,
          ]
            ## information  regarding the edmand
            current_time_ind = ((self.time_varying_demand_model["time_points"] == int(self.current_time)).nonzero(as_tuple=True)[0]).item()
            if self.time_varying_demand_model["time_points"].shape[0] == current_time_ind + 1:
                next_time_ind = current_time_ind
            else:
                next_time_ind = current_time_ind+1
            from_sum = self.time_varying_demand_model["demand"][:, vertiport.id, :].sum(-1)
            to_sum = self.time_varying_demand_model["demand"][:, :, vertiport.id].sum(-1)
            props.extend([(from_sum/from_sum.max())[current_time_ind], (from_sum/from_sum.max())[next_time_ind]]),
            props.extend([(to_sum/to_sum.max())[current_time_ind], (to_sum/to_sum.max())[next_time_ind]])
            # information regarding the passenger price
            # props.extend(self.passenger_pricing_model.view(-1)/self.passenger_pricing_model.view(-1).max())

            Vertiport_graph_nods.append(props)

        vertiport_graph = {}
        vertiport_graph["nodes"] = th.tensor(Vertiport_graph_nods)
        vertiport_graph["adjacency"] = 1/(1+th.cdist(vertiport_graph["nodes"],vertiport_graph["nodes"]))
        for i in range(self.n_vertiports):
            vertiport_graph["adjacency"][i,i] = 0.0
        return vertiport_graph


    def get_evtol_graph(self):

        evtols_graph_nodes = []
        for evtol in self.evtols:
            props = [self.vertiport_locations[evtol.current_location,0].item(),
                     self.vertiport_locations[evtol.current_location, 1].item(),
                     evtol.max_passenger,
                     evtol.landing_time,
                     evtol.take_off_time,
                     evtol.speed,
                     evtol.next_flight_time,
                     evtol.next_decision_time,
                     evtol.battery.battery_max]
            evtols_graph_nodes.append(props)
        evtol_graph = {}
        evtol_graph["nodes"] = th.tensor(evtols_graph_nodes)
        evtol_graph["adjacency"] = 1/(1+th.cdist(evtol_graph["nodes"], evtol_graph["nodes"]))
        for i in range(self.n_evtols):
            evtol_graph["adjacency"][i,i] = 0.0
        return evtol_graph

    def get_demand_graph(self):
        pass

    def get_passenger_price_graph(self):
        pass

    def generate_demand_model(self):
        demand = th.randint(0, 100, (self.n_vertiports, self.n_vertiports))
        for i in range(self.n_vertiports):
            demand[i, i] = 0
        return demand

    def generate_electricity_pricing_model(self):
        price = 1.0
        return th.tensor([price])


    def generate_evtols_starting_locations(self):
        return th.randint(0, self.n_vertiports, (self.n_evtols, 1))

    def generate_vertiports_max_charged(self):
        pass

    def generate_vertiports_max_parked(self):
        pass

    def generate_passenger_pricing_model(self):

        return ((self.vertiports_distance_matrix * .8).to(th.int64)).to(th.float32)

    def single_peak_demand(self, time, peak_time, peak, base, std):
        demand = int((1/(std*1.41*3.14))*np.exp(-.5*((time - peak_time)/std)**2)*peak + base) + th.randint(1,10, (1,)).item()
        return demand

    def double_peak_demand(self, time, peak_time1, peak_time2, peak1, peak2, base, std1, std2):
        if time < (peak_time1 + peak_time2)/2:
            demand = int((1 / (std1 * 1.41 * 3.14)) * np.exp(-.5 * ((time - peak_time1) / std1) ** 2) * peak1 + base)
        else:
            demand = int((1 / (std2 * 1.41 * 3.14)) * np.exp(-.5 * ((time - peak_time2) / std2) ** 2) * peak2 + base)

        return demand + + th.randint(1,10, (1,)).item()

    def get_time_varying_demand(self):
        urban_center = th.tensor([[45.00, 55.00]])
        radius = 30
        urbans = (th.cdist(self.vertiport_locations, urban_center) < radius).nonzero(as_tuple=True)[0]
        time_points = np.linspace(6.00, self.time_horizon, 13)
        demand_data = th.zeros((len(time_points), self.n_vertiports, self.n_vertiports))
        for i in range(len(time_points)):
            time = time_points[i]
            for j in range(self.n_vertiports):
                for k in range(self.n_vertiports):
                    if j != k:
                        if j in urbans and k in urbans:
                            demand_data[i, j, k] = self.double_peak_demand(time, 9.00, 4.30, 100, 100, 40, .5, .5)
                        elif j in urbans and k not in urbans:
                            demand_data[i, j, k] = self.single_peak_demand(time, 4.30, 30, 20, .5)
                        elif j not in urbans and k in urbans:
                            demand_data[i, j, k] = self.single_peak_demand(time, 9.00, 30, 20, .5)
                        else:
                            demand_data[i, j, k] = 20 + th.randint(1, 10, (1,)).item()
        return {"demand":demand_data, "time_points": th.tensor(time_points)}