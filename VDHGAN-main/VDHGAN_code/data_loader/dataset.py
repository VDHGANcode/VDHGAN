import copy
import json
import logging
import sys
import os
os.chdir(sys.path[0])
import torch
from dgl import DGLGraph
import dgl
from tqdm import tqdm
from dgl import graph
from data_loader.batch_graph import GGNNBatchGraph, HANBatchGraph
from utils import load_default_identifiers, initialize_batch, debug


class DataEntry:
    def __init__(self, dataset, num_nodes, features, node_types, edges, target):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.target = target
        self.features = torch.FloatTensor(features)
        self.node_types = torch.tensor(self.dataset.get_node_type_numbers(node_types))
        #self.graph = DGLGraph()
        self.graph = dgl.graph([])
        #self.create_dgl_heterograph(node_types, edges, datset)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features, 'node_types': self.node_types})  ##
        for s, _type, t in edges:
            etype_number = self.dataset.get_edge_type_number(_type)
            # self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])})

    def create_dgl_heterograph(self, node_types, edges, dataset, meta_path=1):
        self.graph = dgl.heterograph(
            {
                ("Expression", "ED", "Declaration"): [],
                ("Declaration", "DE", "Expression"): [],

                ("Expression", "EC", "ControlFlow"): [],
                ("ControlFlow", "CE", "Expression"): [],

                ("Expression", "EO", "Other"): [],
                ("Other", "OE", "Expression"): [],

                ("Declaration", "DC", "ControlFlow"): [],
                ("ControlFlow", "CD", "Declaration"): [],

                ("Declaration", "DO", "Other"): [],
                ("Other", "OD", "Declaration"): [],

                ("ControlFlow", "CO", "Other"): [],
                ("Other", "OC", "ControlFlow"): [],

                ("Expression", "EE", "Expression"): [],
                ("Declaration", "DD", "Declaration"): [],
                ("ControlFlow", "CC", "ControlFlow"): [],
                ("Other", "OO", "Other"): [],
            }
        )
        #self.graph.add_nodes(self.num_nodes, data={'features': self.features})  ##
        for i, ntype in node_types.items():
            #self.graph.add_nodes(1, data={'features': self.features[int(i)]}, ntype=ntype)
            self.graph.add_nodes(1, ntype=ntype)
        for s, _type, t in edges:
            s_type = node_types[str(s)]
            d_type = node_types[str(t)]
            e_type = f"{s_type[0]}{d_type[0]}"

            etype_number = dataset.get_edge_type_number(_type)
            # self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})
            self.graph.add_edges(s, t, data={'etype': torch.LongTensor([etype_number])}, etype=e_type)
        #print(self.graph.number_of_nodes())


class DataSet:
    def __init__(self, train_src, valid_src, test_src, batch_size, n_ident=None, g_ident=None, l_ident=None, node_ident=None, metapath_list=None):
        self.train_examples = []
        self.valid_examples = []
        self.test_examples = []
        self.train_batches = []
        self.valid_batches = []
        self.test_batches = []
        self.batch_size = batch_size
        self.metapath_list = metapath_list
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident, self.g_ident, self.l_ident, self.node_ident= load_default_identifiers(n_ident, g_ident, l_ident, node_ident)
        self.read_dataset(train_src, valid_src, test_src)
        self.initialize_dataset()

    def initialize_dataset(self):

        self.initialize_train_batch()
        self.initialize_valid_batch()
        self.initialize_test_batch()

    def read_dataset(self, train_src, valid_src, test_src):
        debug('Reading Train File!')

        
        with open(train_src,"r") as fp:
            train_data = []
            train_data = json.load(fp)
            #for entry in tqdm(train_data[:70]):
            for entry in tqdm(train_data):
                example = DataEntry(dataset=self, num_nodes=len(entry[self.n_ident]), features=entry[self.n_ident], node_types=entry[self.node_ident], edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                                  
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                    debug('Feature Size %d' % self.feature_size)
                self.train_examples.append(example)
        
        if valid_src is not None:
            debug('Reading Validation File!')
            
            with open(valid_src,"r") as fp:
                valid_data = []
                valid_data = json.load(fp) 
                for entry in tqdm(valid_data):
                    example = DataEntry(dataset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident], node_types=entry[self.node_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.valid_examples.append(example)
        if test_src is not None:
            debug('Reading Test File!')
            with open(test_src) as fp:
                test_data = []
                test_data = json.load(fp)
                for entry in tqdm(test_data):
                    example = DataEntry(dataset=self, num_nodes=len(entry[self.n_ident]),
                                        features=entry[self.n_ident], node_types=entry[self.node_ident],
                                        edges=entry[self.g_ident], target=entry[self.l_ident][0][0])
                    self.test_examples.append(example)


    def get_edge_type_number(self, _type):
        if _type not in self.edge_types:
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    def get_node_type_numbers(self, node_types):
        label_mapping = {'Expression': 0, 'Declaration': 1, 'ControlFlow': 2, 'Other': 3}
        output_list = [label_mapping[label] for label in node_types.values()]
        return output_list

    @property
    def max_edge_type(self):
        return self.max_etype

    def initialize_train_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.train_batches = initialize_batch(self.train_examples, batch_size, shuffle=False)
        return len(self.train_batches)
        pass

    def initialize_valid_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.valid_batches = initialize_batch(self.valid_examples, batch_size, shuffle=False)
        return len(self.valid_batches)
        pass

    def initialize_test_batch(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        self.test_batches = initialize_batch(self.test_examples, batch_size, shuffle=False)
        
        return len(self.test_batches)
        pass

    def get_dataset_by_ids_for_GGNN_ori(self, entries, ids):
        # 从数据集中选取指定索引的数据
        taken_entries = [entries[i] for i in ids]
        # 提取选中数据的标签
        labels = [e.target for e in taken_entries]
        # 创建 GGNNBatchGraph 对象，用于存储批处理的子图信息
        batch_graph = GGNNBatchGraph()
        # 遍历取出的数据，将其子图添加到 batch_graph 中

        for entry in taken_entries:

            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        # 返回批处理的图数据和相应的标签
        return batch_graph, torch.FloatTensor(labels)

    def get_dataset_by_ids_for_GGNN(self, entries, ids):
        # 从数据集中选取指定索引的数据
        taken_entries = [entries[i] for i in ids]
        # 提取选中数据的标签
        labels = [e.target for e in taken_entries]
        # 创建 GGNNBatchGraph 对象，用于存储批处理的子图信息
        batch_graph = HANBatchGraph(metapath_list=self.metapath_list)
        # 遍历取出的数据，将其子图添加到 batch_graph 中

        for entry in taken_entries:
            batch_graph.add_subgraph(copy.deepcopy(entry.graph))
        # 返回批处理的图数据和相应的标签
        #batch_graph.generate_meta_graph()
        #return batch_graph, torch.FloatTensor(labels)
        return batch_graph, torch.FloatTensor(labels)

    def get_dataset_by_ids_for_GGNN_1(self, entries, ids):
        # 从数据集中选取指定索引的数据
        taken_entries = [entries[i] for i in ids]
        # 提取选中数据的标签
        labels = [e.target for e in taken_entries]
        # 创建 GGNNBatchGraph 对象，用于存储批处理的子图信息
        batch_graph = HANBatchGraph(taken_entries[0].graph, taken_entries[0].features, self.metapath_list)
        # 遍历取出的数据，将其子图添加到 batch_graph 中
        # for entry in taken_entries:
        #     batch_graph.add_subgraph(copy.deepcopy(entry.graph))
            # 返回批处理的图数据和相应的标签
            #block_list, h_list, edge_types, labels
        return batch_graph.get_network_inputs(), torch.FloatTensor(labels)

    def get_next_train_batch(self):
        # 如果训练批次列表为空，进行初始化
        if len(self.train_batches) == 0:
            self.initialize_train_batch()
        # 从训练批次列表中取出一批次的数据 ids
        ids = self.train_batches.pop()
        # 调用 get_dataset_by_ids_for_GGNN 获取批处理的图数据和标签
        return self.get_dataset_by_ids_for_GGNN(self.train_examples, ids)

    def get_next_valid_batch(self):
        if len(self.valid_batches) == 0:
            self.initialize_valid_batch()
        ids = self.valid_batches.pop()

        return self.get_dataset_by_ids_for_GGNN(self.valid_examples, ids)

    def get_next_test_batch(self):
        if len(self.test_batches) == 0:
            self.initialize_test_batch()
        ids = self.test_batches.pop()
        return self.get_dataset_by_ids_for_GGNN(self.test_examples, ids)


