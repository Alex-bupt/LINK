import os
import json
import torch
import numpy as np
from scipy.sparse import coo_matrix
from config import args


class DBpediaHandler:
    def __init__(self, dataset):
        self.dataset_dir = os.path.join('data', dataset)
        with open(os.path.join(self.dataset_dir, 'dbpedia_subkg.json'), 'r', encoding='utf-8') as f:
            self.entity_kg = json.load(f)
        with open(os.path.join(self.dataset_dir, 'entity2id.json'), 'r', encoding='utf-8') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(self.dataset_dir, 'item_ids.json'), 'r', encoding='utf-8') as f:
            self.item_ids = json.load(f)

        self.processed_kg()

    def processed_kg(self):
        edge_list = []
        for entity, connections in self.entity_kg.items():
            for relation, tail in connections:
                edge_list.append((self.entity2id[entity], self.entity2id[tail]))

        edge_list = list(set(edge_list))  # 去重
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        self.edge_index = edge_index
        self.num_entities = len(self.entity2id)
        self.item_ids = set(self.item_ids)


class DataHandler:
    def __init__(self):
        self.dbpedia_handler = DBpediaHandler(dataset='redial')
        self.trnMat = self.buildAdjacencyMatrix()
        self.user, self.item = self.trnMat.shape
        self.torchBiAdj = self.makeTorchAdj(self.trnMat)

        self.load_features()

    def buildAdjacencyMatrix(self):
        entity_kg = self.dbpedia_handler.edge_index.numpy()
        rows, cols = entity_kg

        # 为实体创建稀疏邻接矩阵
        data = np.ones(len(rows))
        adj_matrix = coo_matrix((data, (rows, cols)), shape=(self.dbpedia_handler.num_entities, self.dbpedia_handler.num_entities))
        return adj_matrix

    def makeTorchAdj(self, mat):
        mat = self.normalizeAdj(mat)
        idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = torch.from_numpy(mat.data.astype(np.float32))
        shape = torch.Size(mat.shape)
        return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

    def load_features(self):
        # 假设DBpedia没有特征，使用随机嵌入
        entity_dim = 768  # 嵌入维度
        self.entity_feats = torch.rand(self.dbpedia_handler.num_entities, entity_dim).cuda()


if __name__ == '__main__':
    data_handler = DataHandler()
    print(data_handler.dbpedia_handler.num_entities)
    print(data_handler.dbpedia_handler.edge_index)
    print(data_handler.trnMat.shape)
    print(data_handler.torchBiAdj)
    print(data_handler.entity_feats.shape)
    print(data_handler.entity_feats)