"""
Author: Steve Paul 
Date: 1/18/22 """

from torch import nn
import torch

class SimpleNN(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=2
                 ):
        super(SimpleNN, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)


    def forward(self, data, mask=None):

        X = data['location'][:,1:,:]
        # X = torch.cat((data['loc'], data['deadline']), -1)

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)



        # init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = F0#torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class GCAPCNFeatureExtractorNTDA(nn.Module):

    def __init__(self,
                 n_layers=2,
                 n_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=2
                 ):
        super(GCAPCNFeatureExtractorNTDA, self).__init__()
        self.n_layers = n_layers
        self.n_dim = n_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, n_dim * n_p)
        self.init_embed_depot = nn.Linear(2, n_dim)

        self.W_L_1_G1 = nn.Linear(n_dim * (n_K + 1) * n_p, n_dim)

        self.normalization_1 = nn.BatchNorm1d(n_dim * n_p)

        self.W_F = nn.Linear(n_dim * n_p, n_dim)

        self.activ = nn.Tanh()

    def forward(self, data, mask=None):

        X = data['location'][:,1:,:]
        # X = torch.cat((data['loc'], data['deadline']), -1)
        X_loc = data['location'][:,1:,:]
        distance_matrix = ((((X_loc[:, :, None] - X_loc[:, None]) ** 2).sum(-1)) ** .5)
        num_samples, num_locations, _ = X.size()
        A = ((1 / distance_matrix) * (torch.eye(num_locations, device=distance_matrix.device).expand(
            (num_samples, num_locations, num_locations)) - 1).to(torch.bool).to(torch.float))
        A[A != A] = 0
        D = torch.mul(torch.eye(num_locations, device=distance_matrix.device).expand((num_samples, num_locations, num_locations)),
                      (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3
        L = D - A
        L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :],
                                          torch.matmul(L_squared, F0)[:, :, :]
                                          ),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) #+ F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        # init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = F_final#torch.cat((init_depot_embed, F_final), 1)
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )



class GCAPCNEvtolFeatureExtractor(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=1
                 ):
        super(GCAPCNEvtolFeatureExtractor, self).__init__()
        self.n_layers = n_layers
        self.n_dim = features_dim
        self.features_dim=features_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * n_p)

        self.W_L_1_G1 = nn.Linear(features_dim * (n_K + 1) * n_p, features_dim)

        self.normalization_1 = nn.BatchNorm1d(features_dim * n_p)

        self.W_F = nn.Linear(features_dim * n_p, features_dim)


        self.activ = nn.Tanh()

    def forward(self, data):


        X = data['evtol_graph_nodes']
        # distance_matrix = ((((X[:, :, None] - X[:, None]) ** 2).sum(-1)) ** .5)[0]



        num_samples, num_locations, _ = X.size()

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3

        A = data["evtol_graph_adjacency"]
        D = torch.mul(
            torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        # L_topo = data["adjacency"]
        # L = L_topo
        # L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          # torch.matmul(L_squared, F0)[:, :, :]
                                          ),
                                         -1))


        F1 = g_L1_1#torch.cat((g_L1_1), -1)
        F1 = self.activ(F1) #+ F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        # init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = F_final#torch.cat((init_depot_embed, F_final), 1)

        # context = self.full_context_nn(
        #               torch.cat((h.mean(dim=2)[:, None, :], data['agent_taking_decision_coordinates'],
        #                       self.agent_context(data['agents_destination_coordinates']).sum(2)[:, None, :],
        #                         data['mask'].permute(0,2,1)),
        #                      -1))
        # mask_shape = data['mask'].shape
        # if data["first_dec"][0,0] != 1:
        #     h = h.detach()
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

class GCAPCNFeatureExtractor(nn.Module):

    def __init__(self,
                 n_layers=2,
                 features_dim=128,
                 n_p=1,
                 node_dim=2,
                 n_K=1
                 ):
        super(GCAPCNFeatureExtractor, self).__init__()
        self.n_layers = n_layers
        self.n_dim = features_dim
        self.features_dim = features_dim
        self.n_p = n_p
        self.n_K = n_K
        self.node_dim = node_dim
        self.init_embed = nn.Linear(node_dim, features_dim * n_p)

        self.W_L_1_G1 = nn.Linear(features_dim * (n_K + 1) * n_p, features_dim)

        self.normalization_1 = nn.BatchNorm1d(features_dim * n_p)

        self.W_F = nn.Linear(features_dim * n_p, features_dim)

        self.activ = nn.Tanh()

    def forward(self, data):


        X = data['vertiport_graph_nodes']

        # distance_matrix = ((((X[:, :, None] - X[:, None]) ** 2).sum(-1)) ** .5)[0]

        num_samples, num_locations, _ = X.size()

        # Layer 1

        # p = 3
        F0 = self.init_embed(X)

        # K = 3

        A = data["vertiport_graph_adjacency"]

        D = torch.mul(
            torch.eye(num_locations, device=X.device).expand((num_samples, num_locations, num_locations)),
            (A.sum(-1) - 1)[:, None].expand((num_samples, num_locations, num_locations)))
        L = D - A
        # L_topo = data["adjacency"]
        # L = L_topo
        # L_squared = torch.matmul(L, L)
        # L_cube = torch.matmul(L, L_squared)

        g_L1_1 = self.W_L_1_G1(torch.cat((F0[:, :, :],
                                          torch.matmul(L, F0)[:, :, :]
                                          # torch.matmul(L_squared, F0)[:, :, :]
                                          ),
                                         -1))

        F1 = g_L1_1  # torch.cat((g_L1_1), -1)
        F1 = self.activ(F1)  # + F0
        # F1 = self.normalization_1(F1)

        F_final = self.activ(self.W_F(F1))

        # init_depot_embed = self.init_embed_depot(data['depot'])[:]
        h = F_final  # torch.cat((init_depot_embed, F_final), 1)

        # context = self.full_context_nn(
        #               torch.cat((h.mean(dim=2)[:, None, :], data['agent_taking_decision_coordinates'],
        #                       self.agent_context(data['agents_destination_coordinates']).sum(2)[:, None, :],
        #                         data['mask'].permute(0,2,1)),
        #                      -1))
        # mask_shape = data['mask'].shape
        # if data["first_dec"][0,0] != 1:
        #     h = h.detach()
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )