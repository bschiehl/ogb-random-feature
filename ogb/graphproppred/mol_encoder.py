import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims 

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, rfParams):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()
        self.rfParams = rfParams

        for i, dim in enumerate(full_atom_feature_dims):
            if rfParams is None or rfParams['emb']:
                emb = torch.nn.Embedding(dim, emb_dim)
            else:
                emb = torch.nn.Embedding(dim, emb_dim - rfParams['num_rf'])
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
        if rfParams is not None and rfParams['emb']:
            for i in range(rfParams['num_rf']):
                emb = torch.nn.Embedding(rfParams['max_val'], emb_dim)
                self.atom_embedding_list.append(emb)
        
    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            # shape (x.shape[0], emb_dim)
            x_embedding += self.atom_embedding_list[i](x[:,i])
            
        if self.rfParams is not None and not self.rfParams['emb']:
            rand = self.__sample(x.shape[0])
            x_embedding = torch.concat((x_embedding, rand.to(x_embedding.device, x_embedding.dtype)), dim=-1)
            
        return x_embedding

    def __sample(self, num_nodes):
        if self.rfParams['dist'] == "uniform":
            c = (self.rfParams['unif_range'][1] - self.rfParams['unif_range'][0]) * torch.rand((num_nodes, self.rfParams['num_rf']), dtype=torch.float) + self.rfParams['unif_range'][0]
        elif self.rfParams['dist'] == "normal":
            means = torch.full((num_nodes, self.rfParams['num_rf']), self.rfParams['normal_params'][0]).float()
            stds = torch.full((num_nodes, self.rfParams['num_rf']), self.rfParams['normal_params'][1]).float()
            c = torch.normal(means, stds).float()
        else:
            raise ValueError("Invalid distribution")
        mask = torch.rand((num_nodes, self.rfParams['num_rf']), dtype=torch.float).ge(self.rfParams['percent'] / 100)
        z = torch.zeros((num_nodes, self.rfParams['num_rf']))
        c[mask] = z[mask]
        return c




class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i])

        return bond_embedding   


if __name__ == '__main__':
    from loader import GraphClassificationPygDataset
    dataset = GraphClassificationPygDataset(name = 'tox21')
    atom_enc = AtomEncoder(100)
    bond_enc = BondEncoder(100)

    print(atom_enc(dataset[0].x))
    print(bond_enc(dataset[0].edge_attr))




