from typing import List, Optional, Union, Tuple

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
            "normal": normal distribution, "uniform": uniform distribution on [a, b]
            (default: "normal")
        normal_parameters (tuple(float, float), optional): Mean and standard deviation of 
            the normal distribution. (default: :obj:`(0,1)`)
        unif_range (tuple(float, float), optional): Range of the uniform distribution.
            (default: :obj:`(-1,1)`)
        cat (bool, optional): If set to :obj:`False`, existing node features
            will be replaced. (default: :obj:`True`)
        node_types (str or List[str], optional): The specified node type(s) to
            append constant values for if used on heterogeneous graphs.
            If set to :obj:`None`, constants will be added to each node feature
            :obj:`x` for all existing node types. (default: :obj:`None`)
        max_val (int, optional): Value to cap random values at.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        percent: float = 100.0,
        dist: str = "normal",
        normal_parameters: Tuple[float, float] = (0.0,1.0),
        unif_range: Tuple[float, float] = (-1.0, 1.0),
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
        max_val: int = None
    ):
        if isinstance(node_types, str):
            node_types = [node_types]

        self.percent = percent
        self.dist = dist
        self.normal_parameters = normal_parameters
        self.unif_range = unif_range
        self.cat = cat
        self.node_types = node_types
        self.max_val = max_val

    def __call__(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:

        for store in data.node_stores:
            if self.node_types is None or store._key in self.node_types:
                num_nodes = store.num_nodes
                if (self.dist == "uniform"):
                    c = (self.unif_range[1] - self.unif_range[0]) * torch.rand((num_nodes, 1), dtype=torch.float) + self.unif_range[0]
                elif(self.dist == "normal"):
                    means = torch.full((num_nodes, 1), self.normal_parameters[0]).float()
                    stds = torch.full((num_nodes, 1), self.normal_parameters[1]).float()
                    c = torch.normal(means, stds).float()
                else:
                    raise ValueError("Invalid distribution")
                c = c % self.max_val

                if hasattr(store, 'x') and self.cat:
                    mask = torch.rand((num_nodes, 1), dtype=torch.float).ge(self.percent / 100)
                    x = store.x.view(-1, 1) if store.x.dim() == 1 else store.x
                    s = torch.sum(x, 1).view(-1,1).float()
                    # another possibility: map [A,B] --> [a,b] with (x - A) * (b-a)/(B-A) + a
                    # A = min(s), B = max(s)
                    s = s % self.max_val
                    c[mask] = s[mask]
                    store.x = torch.cat([x, c.to(x.device, x.dtype)], dim=-1)
                else:
                    store.x = c

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(distribution={self.dist})'