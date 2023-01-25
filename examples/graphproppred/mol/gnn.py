import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from .conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean
import warnings

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean", rfParams=None, gnn2 = False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.gnn2 = gnn2
        self.rfParams = rfParams

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if self.gnn2:
            self.gnn_node_1 = GNN_node(num_layer, 16, JK=JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, rfParams=rfParams, gnn2=True)
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, rfParams=rfParams)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, rfParams=rfParams)
        
        


        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data, batched_trans = None):
        #print(batched_data)
        if self.gnn2:
            h_node_1 = self.gnn_node_1(batched_trans)
            h_node_1 = torch.abs(h_node_1)
            h_node_1 = h_node_1 % self.rfParams['max_val']
            batched_data.x = torch.cat((batched_data.x, h_node_1.to(batched_data.x.device, batched_data.x.dtype)), dim=1)
        
        for i in range(batched_data.x.shape[1]):
            maxi = torch.max(batched_data.x[:,i])
            mini = torch.min(batched_data.x[:,i])
            if mini < 0:
                warnings.warn(f"negative value detected: {mini} in col {i}")
            if i == 0 and maxi >= 119:
                warnings.warn(f"value in col {i} bigger than 118")
            elif i == 1 and maxi >= 4:
                warnings.warn(f"value in col {i} bigger than 3")
            elif (i == 2 or i == 3) and maxi >= 12:
                warnings.warn(f"value in col {i} bigger than 11")
            elif i == 4 and maxi >= 10:
                warnings.warn(f"value in col {i} bigger than 9")
            elif (i == 5 or i == 6) and maxi >= 6:
                warnings.warn(f"value in col {i} bigger than 5")
            elif (i == 7 or i == 8) and maxi >= 2:
                warnings.warn(f"value in col {i} bigger than 1") 
            elif (maxi >= 100):
                warnings.warn(f"value in col {i} bigger than 99")
            elif (i >= 25):
                warnings.warn("too many columns")
        

        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)


if __name__ == '__main__':
    GNN(num_tasks = 10)