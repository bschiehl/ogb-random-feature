from typing import List, Optional, Union

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('random')
class RandomFeature(BaseTransform):
    r"""Appends a random value to some percentage of node features :obj:`x`
    (functional name: :obj:`random`).
    Args:
        percent (float, optional): percentage of nodes whose features should be extended by a random value. 
            The remaining features are extended deterministically by the sum of the original features.
            (default: :obj:`100.0`)
        dist (str, optional): Distribution to sample the random values from. Options: 
            "normal": standard normal distribution, "uniform": uniform distribution on [-1, 1]
            (default: "normal")
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
    """
    def __init__(
        self,
        percent: float = 100.0,
        dist: str = "normal",
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.percent = percent
        self.dist = dist
        self.cat = cat
        self.node_types = node_types

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                if (self.dist == "uniform"):
                    c = 2 * torch.rand((num_nodes, 1), dtype=torch.float) - 1
                else:
                    c = torch.randn((num_nodes, 1), dtype=torch.float)

                if hasattr(store, 'x') and self.cat:
                    mask = torch.rand((num_nodes, 1), dtype=torch.float).ge(self.percent / 100)
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    s = torch.sum(x, 1).view(-1,1).float()
                    c[mask] = s[mask]
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distribution={self.dist})'