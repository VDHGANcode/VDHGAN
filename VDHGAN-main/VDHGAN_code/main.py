import argparse
import logging
import os
import pickle
import sys

os.chdir(sys.path[0])

import numpy as np
import torch
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel

import dgl
from data_loader.batch_graph import HANSampler
from modules.model import HAN
from utils import EarlyStopping
from trainer import train
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from utils import tally_param, debug, set_logger, RAdam

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
import math
from torch.optim.optimizer import Optimizer, required

import torch.optim

def Devign_main():
    parser = argparse.ArgumentParser("mini-batch HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    # parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_neighbors", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_heads", type=list, default=[8])
    parser.add_argument("--hidden_units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="Devign_VDHGAN_11")
    # parser.add_argument("--dataset", type=str, default="ACM")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    # parser.add_argument('--dataset', type=str, help='Name of the dataset for experiment.', default='FFmpeg_GS_2')
    parser.add_argument('--input_dir', type=str, help='Input Directory of the parser',
                        default='../devign_dataset/devign_cpg_c2_2_2')
    parser.add_argument('--log_dir', default='devign_VDHGAN.log', type=str)

    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--node_type_tag', type=str, help='Name of the node type.', default='node_type')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    #parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=64)  # default=64
    args = parser.parse_args()
    #args = parser.parse_args().__dict__

    model_dir = os.path.join('models', args.dataset)
    #metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"], ["DC", "CD"], ["DO", "OD"], ["CO", "OC"]]
    metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"],
                     ["DE", "ED"], ["DO", "OD"], ["DC", "CD"],
                     ["CE", "EC"], ["CD", "DC"], ["CO", "OC"],
                     ["OE", "EO"], ["OD", "DO"], ["OC", "CO"]]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)

    input_dir = '../devign_dataset/devign_HAN_ECD'
    # dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v0.json'),
    #                   valid_src=os.path.join(input_dir, 'devign-valid-v0.json'),
    #                   test_src=os.path.join(input_dir, 'devign-test-v0.json'),
    #                   batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
    #                   l_ident=args.label_tag, metapath_list=metapath_list)

    processed_data_path = os.path.join(input_dir, 'devign.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        #debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v0.json'),
                          valid_src=os.path.join(input_dir, 'devign-valid-v0.json'),
                          test_src=os.path.join(input_dir, 'devign-test-v0.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag, metapath_list=metapath_list)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples),
                 len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches),
                 len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)

    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    # model = DevignModel(input_dim=dataset.feature_size, output_dim=100,  # output_dim=100
    #                     num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    model = HAN(
        num_metapath=len(metapath_list),
        in_size=dataset.feature_size,
        hidden_size=args.hidden_units,
        out_size=64,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)

    #debug('Total Parameters : %d' % tally_param(model))
    #debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    model.cuda()
    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 1.2])).float(), reduction='sum')
    loss_function.cuda()
    #LR = 1e-4

    #optim = RAdam(model.parameters(), lr=LR, weight_decay=1e-6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )


    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),  # epoches=100
          loss_function=loss_function, optimizer=optimizer,
          save_path=model_dir + '/HAN_Model', max_patience=5, args=args)


def Reveal_main():
    parser = argparse.ArgumentParser("mini-batch HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")

    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--num_heads", type=list, default=[8])
    parser.add_argument("--hidden_units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="Reveal_VDHGAN_test")
    parser.add_argument('--log_dir', default='Reveal_VDHGAN_test.log', type=str)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--node_type_tag', type=str, help='Name of the node type.', default='node_type')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    # parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=32)  # default=64
    args = parser.parse_args()
    # args = parser.parse_args().__dict__

    model_dir = os.path.join('models', args.dataset)
    # metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"], ["DC", "CD"], ["DO", "OD"], ["CO", "OC"]]
    metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"],
                     ["DE", "ED"], ["DO", "OD"], ["DC", "CD"],
                     ["CE", "EC"], ["CD", "DC"], ["CO", "OC"],
                     ["OE", "EO"], ["OD", "DO"], ["OC", "CO"]]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)

    input_dir = '../Reveal_dataset/Reveal_HAN_ECD'
    # dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v0.json'),
    #                   valid_src=os.path.join(input_dir, 'devign-valid-v0.json'),
    #                   test_src=os.path.join(input_dir, 'devign-test-v0.json'),
    #                   batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
    #                   l_ident=args.label_tag, metapath_list=metapath_list)

    processed_data_path = os.path.join(input_dir, 'Reveal.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        # debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, 'Reveal-train-v1.json'),
                          valid_src=os.path.join(input_dir, 'Reveal-valid-v1.json'),
                          test_src=os.path.join(input_dir, 'Reveal-test-v1.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag, metapath_list=metapath_list)
        # dataset = DataSet(train_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-train-v2.json'),
        # valid_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-valid-v2.json'),
        # test_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-test-v2.json'),
        # batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
        # l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples),
                 len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches),
                 len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)

    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    # model = DevignModel(input_dim=dataset.feature_size, output_dim=100,  # output_dim=100
    #                     num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    model = HAN(
        num_metapath=len(metapath_list),
        in_size=dataset.feature_size,
        hidden_size=args.hidden_units,
        out_size=64,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)


    # debug('Total Parameters : %d' % tally_param(model))
    # debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    model.cuda()
    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 8.5])).float(), reduction='sum')
    loss_function.cuda()
    # LR = 1e-4

    # optim = RAdam(model.parameters(), lr=LR, weight_decay=1e-6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),  # epoches=100
            loss_function=loss_function, optimizer=optimizer,
            save_path=model_dir + '/HAN_Model', max_patience=20, args=args)



def Fan_main():
    parser = argparse.ArgumentParser("mini-batch HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")

    parser.add_argument("--num_neighbors", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_heads", type=list, default=[8])
    parser.add_argument("--hidden_units", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="Fan_VDHGAN_2")
    parser.add_argument('--log_dir', default='Fan_VDHGAN_2.log', type=str)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--node_type_tag', type=str, help='Name of the node type.', default='node_type')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='targets')

    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    # parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)  # default=64
    args = parser.parse_args()
    # args = parser.parse_args().__dict__

    model_dir = os.path.join('models', args.dataset)
    # metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"], ["DC", "CD"], ["DO", "OD"], ["CO", "OC"]]
    metapath_list = [["ED", "DE"], ["EC", "CE"], ["EO", "OE"],
                     ["DE", "ED"], ["DO", "OD"], ["DC", "CD"],
                     ["CE", "EC"], ["CD", "DC"], ["CO", "OC"],
                     ["OE", "EO"], ["OD", "DO"], ["OC", "CO"]]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    log_dir = os.path.join(model_dir, args.log_dir)
    set_logger(log_dir)

    input_dir = '../Fan_dataset/Fan_HAN_ECD'
    # dataset = DataSet(train_src=os.path.join(input_dir, 'devign-train-v0.json'),
    #                   valid_src=os.path.join(input_dir, 'devign-valid-v0.json'),
    #                   test_src=os.path.join(input_dir, 'devign-test-v0.json'),
    #                   batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
    #                   l_ident=args.label_tag, metapath_list=metapath_list)

    processed_data_path = os.path.join(input_dir, 'Fan_small.bin')
    logging.info('#' * 100)
    if True and os.path.exists(processed_data_path):
        # debug('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        logging.info('Reading already processed data from %s!' % processed_data_path)
    else:
        logging.info('Loading the dataset from %s' % input_dir)
        dataset = DataSet(train_src=os.path.join(input_dir, 'Fan-train-v1.json'),
                          valid_src=os.path.join(input_dir, 'Fan-valid-v1.json'),
                          test_src=os.path.join(input_dir, 'Fan-test-v1.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag, metapath_list=metapath_list)
        # dataset = DataSet(train_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-train-v2.json'),
        # valid_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-valid-v2.json'),
        # test_src=os.path.join(input_dir, './devign_cpg_v2_1/devign-test-v2.json'),
        # batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
        # l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()

    logging.info('train_dataset: %d; valid_dataset: %d; test_dataset: %d', len(dataset.train_examples),
                 len(dataset.valid_examples), len(dataset.test_examples))
    logging.info("train_batch: %d, valid_batch: %d, test_batch: %d", len(dataset.train_batches),
                 len(dataset.valid_batches), len(dataset.test_batches))
    logging.info('#' * 100)

    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    # model = DevignModel(input_dim=dataset.feature_size, output_dim=100,  # output_dim=100
    #                     num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    model = HAN(
        num_metapath=len(metapath_list),
        in_size=dataset.feature_size,
        hidden_size=args.hidden_units,
        out_size=64,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)


    # debug('Total Parameters : %d' % tally_param(model))
    # debug('#' * 100)
    logging.info('Total Parameters : %d' % tally_param(model))
    logging.info('#' * 100)
    model.cuda()
    loss_function = CrossEntropyLoss(weight=torch.from_numpy(np.array([1, 26])).float(), reduction='sum')
    loss_function.cuda()
    # LR = 1e-4

    # optim = RAdam(model.parameters(), lr=LR, weight_decay=1e-6)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train(model=model, dataset=dataset, epoches=100, dev_every=len(dataset.train_batches),  # epoches=100
            loss_function=loss_function, optimizer=optimizer,
            save_path=model_dir + '/HAN_Model', max_patience=20, args=args)

if __name__ == '__main__':
    Devign_main()
    #Reveal_main()
    #Fan_main()






