import dgl
import torch
from dgl import DGLGraph
import torch as th
import torch.nn.functional as B
import dgl.convert as convert
from dgl import backend as F

from dgl._ffi.function import _init_api
import numpy as np
from dgl.sampling.randomwalks import random_walk
from dgl import utils
from dgl.sampling.pinsage import _select_pinsage_neighbors


class HomogeneousGraphFromPaths(object):
    def __init__(self, G, metapath, num_neighbors, weight_column='weights'):
        self.G = G
        self.weight_column = weight_column
        self.ntype = G.to_canonical_etype(metapath[0])[0]
        self.num_neighbors = num_neighbors

        if metapath is None:
            if len(G.ntypes) > 1 or len(G.etypes) > 1:
                raise ValueError("Metapath must be specified if the graph is homogeneous.")
            metapath = [G.to_canonical_etype[0]]
        self.metapath_hops = len(metapath)
        self.metapath = metapath
        self.full_metapath = metapath

        restart_prob = np.zeros(self.metapath_hops)
        restart_prob[self.metapath_hops::self.metapath_hops] = 0
        restart_prob = F.tensor(restart_prob, dtype=F.float32)
        self.restart_prob = F.copy_to(restart_prob, G.device)


    def __call__(self, seed_nodes):
        """
        Parameters
        ----------
        paths : Tensor
            A tensor representing the paths. Each row in the tensor corresponds to a path,
            and each element in a row is a node ID.

        Returns
        -------
        g : DGLGraph
            A homogeneous graph constructed by selecting neighbors for each given node according
            to the algorithm above.
        """

        #all_nodes = th.arange(1, self.G.number_of_nodes(self.ntype))
        seed_nodes = utils.prepare_tensor(self.G, seed_nodes, 'seed_nodes')

        #paths = F.toindex(paths)
        paths, _ = random_walk(self.G, seed_nodes, metapath=self.full_metapath, restart_prob=self.restart_prob)

        # Extract source and destination nodes from paths
        src = F.reshape(paths[:, self.metapath_hops::self.metapath_hops], (-1,))
        dst = F.repeat(paths[:, 0], self.metapath_hops, 0)

        # Use the _select_pinsage_neighbors function to get counts and selected nodes
        src, dst, counts = _select_pinsage_neighbors(
            src, dst, paths.shape[0], self.num_neighbors)  # Assuming num_neighbors=1

        # Convert to a homogeneous graph
        neighbor_graph = convert.heterograph(
            {(self.ntype, '_E', self.ntype): (src, dst)},
            {self.ntype: self.G.number_of_nodes(self.ntype)}
        )

        # Add edge weights to the graph
        neighbor_graph.edata[self.weight_column] = counts

        #Add self loop
        neighbor_graph = dgl.remove_self_loop(neighbor_graph)
        #neighbor_graph.add_edges(torch.tensor(seed_nodes), torch.tensor(seed_nodes))
        neighbor_graph.add_edges(seed_nodes.clone().detach(), seed_nodes.clone().detach())

        return neighbor_graph

# Example usage
# G = ...  # Your heterogeneous graph
# metapath = ...  # Your metapath
# paths = ...  # Your tensor of paths
#
# # Create the HomogeneousGraphFromPaths instance
# homogeneous_graph_builder = HomogeneousGraphFromPaths(G, metapath)
#
# # Call the instance to get the homogeneous graph
# homogeneous_graph = homogeneous_graph_builder(paths)