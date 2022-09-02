# -*- coding: utf-8 -*-
# @Time     : 2022/2/16 9:50
# @Author   : xiehl
# @Software : PyCharm
# ---------------------------
import numpy as np
import pandas as pd
import json
'''邻接矩阵法完成图的表示'''

#创建图，输入图的顶点个数、顶点、以及创建邻接表和存储顶点的数组
class Graph(object):

    def __init__(self, files_list: list, max_hop=1, dilation=1):

        self.max_hop = max_hop
        self.dilation = dilation
        self.files_list = files_list
        self.get_edge()

        self.hop_dis = self.get_hop_distance(self.num_node, self.edge)
        """最终A的第一维度为2，是因为以当前点为根节点，故有两套不同的权重向量，进一步想表明不同节点的关系"""
        self.get_adjacency()


    def get_edge(self):

        states_ = ["低", "高"]
        edge_dict = {}
        """
        前8个分别代表：
        0、1：pdcp_mean
        2、3：pdcp_std
        4、5：cqi_mean
        6、7：cqi_std
        """
        for i in range(84):
            if i % 2 == 0:
                edge_dict[i] = states_[0]
            else:
                edge_dict[i] = states_[1]

        for file_path in self.files_list[:-1]:
            data = pd.read_csv(file_path, index_col=0)
            data_ = data.values.tolist()
            for singe_data in data_:
                i += 1
                if file_path == "../../data/neo4j/descrip.csv":
                    print("{}:{}".format(i-41, singe_data[0]))
                edge_dict[i] = singe_data[0]
        # # 将节点写入json文件中
        # jsObj = json.dumps(edge_dict, ensure_ascii=False)  # indent参数是换行和缩进
        # fileObject = open('../../data/real_data/edge_dict.json', 'w', encoding='utf-8')
        # fileObject.write(jsObj)
        # fileObject.close()

        print("一共有{}个节点".format(len(edge_dict)))
        self.num_node = len(edge_dict)

        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_1base = []
        kpi_data = pd.read_csv(self.files_list[-1], index_col=0)

        def find_key(edge_dict, value):
            return [key for key, value_ in edge_dict.items() if value_ == value]

        for index in range(len(kpi_data)):
            col_data = kpi_data.iloc[index, :]

            key_descrip = find_key(edge_dict, col_data["new_abnor_descrip"])[0]
            key_method = find_key(edge_dict, col_data["new_treat_method"])[0]

            index_kpi = kpi_data.columns[3:-3].tolist()

            key_list_ = [[find_key(edge_dict, col_data[rol])[0], find_key(edge_dict, col_data["new_abnor_descrip"])[0]] for rol in ["new_abnor_reason"] + index_kpi]
            add_ = [(0, 0)] + [(x, 0) for x in range(0, len(index_kpi), 2)]
            for i in range(len(add_)):
                key_list_[i][0], key_list_[i][1] = key_list_[i][0]+add_[i][0], key_list_[i][1]+add_[i][1]

            neighbor_1base.extend([tuple(x) for x in key_list_]+[(key_descrip, key_method)])
        self.edge = list(set(self_link + neighbor_1base))
        print("有{}个顶点对".format(len(self.edge)))

    # 获得邻接矩阵
    def get_adjacency(self):

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = self.normalize_digraph(adjacency)

        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
        self.A = A


    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD

    def get_hop_distance(self, num_node, edge, max_hop=1):
        A = np.zeros((num_node, num_node))
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1
        # compute hop steps
        hop_dis = np.zeros((num_node, num_node)) + np.inf
        transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]# 对角连乘（=0），矩阵连乘（>0)
        arrive_mat = (np.stack(transfer_mat) > 0)
        for d in range(max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
        return hop_dis

if __name__ == '__main__':
    descrip_path = "../../data/neo4j/descrip.csv"
    reason_path = "../../data/neo4j/reason.csv"
    method_path = "../../data/neo4j/method.csv"
    kpi_path = "../../data/neo4j/kpi_data.csv"
    files_list = [descrip_path, reason_path, method_path, kpi_path]
    s = Graph(files_list=files_list)
    # s[2, 245, 245]
    print(len(s.A[0][0]))



'''
输入图的顶点的个数:3
输入顶点:a
输入顶点:b
输入顶点:c
输入顶点之间的关系
输入顶点a--b之间的关系(0表示无连通，1表示有连通)1
输入顶点a--c之间的关系(0表示无连通，1表示有连通)1
输入顶点b--c之间的关系(0表示无连通，1表示有连通)0
[0, 1, 1]
[1, 0, 0]
[1, 0, 0]

'''

