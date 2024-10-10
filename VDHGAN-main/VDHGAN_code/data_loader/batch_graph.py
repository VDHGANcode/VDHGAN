import torch
from dgl import DGLGraph
import numpy
import dgl
import dgl.function as fn
from torch import nn
from .HomogeneousGraphFromPaths import HomogeneousGraphFromPaths

# class BatchGraph:
#     def __init__(self, g, f):
#         self.graph = g
#         self.features = f
#         self.number_of_nodes = 0
#         self.graphid_to_nodeids = {}
#         self.num_of_subgraphs = 0

class BatchGraph:
    def __init__(self):
        #self.graph = DGLGraph()
        self.graph = dgl.graph([])
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g):
        #assert isinstance(_g, DGLGraph)
        assert isinstance(_g, DGLGraph)

        num_new_nodes = _g.number_of_nodes()

        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))

        self.graph.add_nodes(num_new_nodes, data=_g.ndata)

        sources, dests = _g.all_edges()

        sources += self.number_of_nodes

        dests += self.number_of_nodes

        self.graph.add_edges(sources, dests, data=_g.edata)

        self.number_of_nodes += num_new_nodes

        self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)

        features = torch.cat([features, torch.zeros(size=(self.number_of_nodes - features.shape[0], features.shape[1]),
                                                   requires_grad=features.requires_grad, device=features.device)], dim=0)
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]

        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)

        return output_vectors#, lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')

from scipy import sparse as sp

class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()
    def get_network_inputs(self, cuda=False, device=None):
        
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']

        if cuda:
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
        pass


class HANBatchGraph(BatchGraph):
    #def __init__(self, g, f, metapath_list=None, num_neighbors=20):
    def __init__(self, metapath_list=None, num_neighbors=20):
        #super(HANBatchGraph, self).__init__(g, f)
        super(HANBatchGraph, self).__init__()
        if not metapath_list:
            raise "metapath_list is NONE"
            #self.metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"], ["DC", "CD"], ["DO", "OD"], ["CO", "OC"]]
        self.metapath_list = metapath_list
        self.num_neighbors = num_neighbors
        #self.all_features = []
        #self.sampler = HANSampler(self.graph, metapath_list, num_neighbors)

    # def add_sub_features(self, features):
    #     self.all_features.append(features)

    def get_network_inputs(self, cuda=False):
        #features = self.graph.ndata['features']
        features = self.features
        edge_types = self.graph.edata['etype']
        sampler = HANSampler_2(self.graph, self.metapath_list, self.num_neighbors)
        #print(self.graph.nodes)

        block_list = sampler.sample_blocks()
        #print(block_list)
        h_list = load_subtensors(block_list, features)

        return block_list, h_list, edge_types

    def create_dgl_heterograph(self, node_types, edges, meta_path=1):
        if meta_path is None:
            raise "meta_path is None!"
        ED = []
        DE = []
        EC = []
        CE = []
        EO = []
        OE = []
        DC = []
        CD = []
        DO = []
        OD = []
        CO = []
        OC = []
        EE = []
        DD = []
        CC = []
        OO = []
        cout_loss = []

        # 添加边
        for src_node, edge_type, dst_node in edges:
            s_type = node_types[src_node]
            d_type = node_types[dst_node]
            if s_type == "Expression" and d_type == "Declaration":
                ED.append((src_node, dst_node))
            elif s_type == "Declaration" and d_type == "Expression":
                DE.append((src_node, dst_node))

            elif s_type == "Expression" and d_type == "ControlFlow":
                EC.append((src_node, dst_node))
            elif d_type == "Expression" and s_type == "ControlFlow":
                CE.append((src_node, dst_node))

            elif s_type == "Expression" and d_type == "Other":
                EO.append((src_node, dst_node))
            elif d_type == "Expression" and s_type == "Other":
                OE.append((src_node, dst_node))

            elif s_type == "Declaration" and d_type == "ControlFlow":
                DC.append((src_node, dst_node))
            elif d_type == "Declaration" and s_type == "ControlFlow":
                CD.append((src_node, dst_node))

            elif s_type == "Declaration" and d_type == "Other":
                DO.append((src_node, dst_node))
            elif d_type == "Declaration" and s_type == "Other":
                OD.append((src_node, dst_node))

            elif s_type == "ControlFlow" and d_type == "Other":
                CO.append((src_node, dst_node))
            elif d_type == "ControlFlow" and s_type == "Other":
                OC.append((src_node, dst_node))

            elif s_type == d_type == "Expression":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "ControlFlow":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "Declaration":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "Other":
                EE.append((src_node, dst_node))
            else:
                cout_loss.append((src_node, edge_type, dst_node))
        #self.dglgraph.add_edges(src_nodes, dst_nodes, {'rel_type': edge_types})
        dglgraph = dgl.heterograph(
            {
                ("Expression", "ED", "Declaration"): ED,
                ("Declaration", "DE", "Expression"): DE,

                ("Expression", "EC", "ControlFlow"): EC,
                ("ControlFlow", "CE", "Expression"): CE,

                ("Expression", "EO", "Other"): EO,
                ("Other", "OE", "Expression"): OE,

                ("Declaration", "DC", "ControlFlow"): DC,
                ("ControlFlow", "CD", "Declaration"): CD,

                ("Declaration", "DO", "Other"): DO,
                ("Other", "OD", "Declaration"): OD,

                ("ControlFlow", "CO", "Other"): CO,
                ("Other", "OC", "ControlFlow"): OC,

                #删去结点自环
                ("Expression", "EE", "Expression"): EE,
                ("Declaration", "DD", "Declaration"): DD,
                ("ControlFlow", "CC", "ControlFlow"): CC,
                ("Other", "OO", "Other"): OO,
            },
            {"Expression": self.number_of_nodes,
             "Declaration": self.number_of_nodes,
             "ControlFlow": self.number_of_nodes,
             "Other": self.number_of_nodes}
        )
        return dglgraph

    def create_dgl_heterograph_2(self, node_types, edges, meta_path=1):
        if meta_path is None:
            raise "meta_path is None!"
        ED = []
        DE = []
        EC = []
        CE = []
        EO = []
        OE = []
        DC = []
        CD = []
        DO = []
        OD = []
        CO = []
        OC = []
        EE = []
        DD = []
        CC = []
        OO = []
        cout_loss = []

        # 添加边
        for src_node, edge_type, dst_node in edges:
            s_type = node_types[src_node]
            d_type = node_types[dst_node]
            if s_type == "Expression" and d_type == "Declaration":
                ED.append((src_node, dst_node))
            elif s_type == "Declaration" and d_type == "Expression":
                DE.append((src_node, dst_node))

            elif s_type == "Expression" and d_type == "ControlFlow":
                EC.append((src_node, dst_node))
            elif d_type == "Expression" and s_type == "ControlFlow":
                CE.append((src_node, dst_node))

            elif s_type == "Expression" and d_type == "Other":
                EO.append((src_node, dst_node))
            elif d_type == "Expression" and s_type == "Other":
                OE.append((src_node, dst_node))

            elif s_type == "Declaration" and d_type == "ControlFlow":
                DC.append((src_node, dst_node))
            elif d_type == "Declaration" and s_type == "ControlFlow":
                CD.append((src_node, dst_node))

            elif s_type == "Declaration" and d_type == "Other":
                DO.append((src_node, dst_node))
            elif d_type == "Declaration" and s_type == "Other":
                OD.append((src_node, dst_node))

            elif s_type == "ControlFlow" and d_type == "Other":
                CO.append((src_node, dst_node))
            elif d_type == "ControlFlow" and s_type == "Other":
                OC.append((src_node, dst_node))

            elif s_type == d_type == "Expression":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "ControlFlow":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "Declaration":
                EE.append((src_node, dst_node))
            elif s_type == d_type == "Other":
                EE.append((src_node, dst_node))
            else:
                cout_loss.append((src_node, edge_type, dst_node))
        # self.dglgraph.add_edges(src_nodes, dst_nodes, {'rel_type': edge_types})
        dglgraph = dgl.heterograph(
            {
                ("Expression", "ED", "Declaration"): ED,
                ("Declaration", "DE", "Expression"): DE,

                ("Expression", "EC", "ControlFlow"): EC,
                ("ControlFlow", "CE", "Expression"): CE,

                ("Expression", "EO", "Other"): EO,
                ("Other", "OE", "Expression"): OE,

                ("Declaration", "DC", "ControlFlow"): DC,
                ("ControlFlow", "CD", "Declaration"): CD,

                ("Declaration", "DO", "Other"): DO,
                ("Other", "OD", "Declaration"): OD,

                ("ControlFlow", "CO", "Other"): CO,
                ("Other", "OC", "ControlFlow"): OC,

                # 删去结点自环
                ("Expression", "EE", "Expression"): EE,
                ("Declaration", "DD", "Declaration"): DD,
                ("ControlFlow", "CC", "ControlFlow"): CC,
                ("Other", "OO", "Other"): OO,
            }
        )
        return dglgraph

    def generate_meta_graph(self, cuda=False):
        # features = self.graph.ndata['features']
        mapping = {
            0: "Expression",
            1: "Declaration",
            2: "ControlFlow",
            3: "Other"
        }

        features = self.graph.ndata['features']
        node_types = self.graph.ndata['node_types'].tolist()
        node_types = [mapping[val] for val in node_types]
        edge_types = self.graph.edata['etype']
        src, dst = self.graph.all_edges()
        edge_list = list(zip(src.tolist(), edge_types.tolist(), dst.tolist()))
        graph = self.create_dgl_heterograph(node_types, edge_list, self.metapath_list)
        #num = graph.number_of_nodes()
        #num2 = graph.number_of_edges()

        sampler = HANSampler_2(graph, self.metapath_list, self.num_neighbors)
        # print(self.graph.nodes)

        block_list = sampler.sample_blocks()
        # print(block_list)
        h_list = load_subtensors(block_list, features)

        return block_list, h_list, edge_types



class HANSampler(object):
    def __init__(self, g, metapath_list, num_neighbors):
        # 使用图 'g'、元路径列表和邻居数量初始化 HANSampler 对象。
        self.sampler_list = []
        for metapath in metapath_list:
            # 为元路径列表中的每个元路径创建 RandomWalkNeighborSampler。
            # 该采样器基于指定的元路径在图上生成随机游走。
            self.sampler_list.append(
                dgl.sampling.RandomWalkNeighborSampler(
                    G=g,
                    num_traversals=1,
                    termination_prob=0,
                    num_random_walks=num_neighbors,
                    num_neighbors=num_neighbors,
                    metapath=metapath,
                )
            )

    def sample_blocks(self, seeds):
        # 给定一组种子节点，使用随机游走采样节点块及其连接。
        block_list = []
        for sampler in self.sampler_list:
            # 使用每个 RandomWalkNeighborSampler 生成围绕种子节点的前沿节点。
            frontier = sampler(seeds)
            # add self loop
            # 从前沿中删除自环并向节点添加显式的自环。
            frontier = dgl.remove_self_loop(frontier)
            frontier.add_edges(torch.tensor(seeds), torch.tensor(seeds))
            # 将前沿转换为 DGL 块结构。
            block = dgl.to_block(frontier, seeds)
            block_list.append(block)

        return seeds, block_list


class HANSampler_2(object):
    def __init__(self, g, metapath_list, num_neighbors):
        # 使用图 'g'、元路径列表和邻居数量初始化 HANSampler 对象。
        self.G = g
        self.sampler_list = []
        for metapath in metapath_list:
            # note: random walk may get same route(same edge), which will be removed in the sampled graph.
            # So the sampled graph's edges may be less than num_random_walks(num_neighbors).
            # 为元路径列表中的每个元路径创建 RandomWalkNeighborSampler。
            # 该采样器基于指定的元路径在图上生成随机游走。
            self.sampler_list.append(
                HomogeneousGraphFromPaths(
                    G=g,
                    metapath=metapath,
                    num_neighbors=num_neighbors,
                )
            )
    def sample_blocks(self):
        # 给定一组种子节点，使用随机游走采样节点块及其连接。
        block_list = []
        for sampler in self.sampler_list:
            # 使用每个 RandomWalkNeighborSampler 生成围绕种子节点的前沿节点。
            all_nodes = torch.arange(1, self.G.number_of_nodes(sampler.ntype))
            frontier = sampler(all_nodes)
            # add self loop
            # 从前沿中删除自环并向节点添加显式的自环。
            #frontier = dgl.remove_self_loop(frontier)
            #frontier.add_edges(torch.tensor(), torch.tensor())
            # 将前沿转换为 DGL 块结构。
            block = dgl.to_block(frontier)
            block_list.append(block)

        return block_list


def load_subtensors(blocks, features):
    h_list = []
    for block in blocks:
        input_nodes = block.srcdata[dgl.NID]
        h_list.append(features[input_nodes])
    return h_list


