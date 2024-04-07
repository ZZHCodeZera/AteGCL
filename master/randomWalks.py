import datetime
import pickle
import random

import numpy as np
import pandas as pd
import networkx as nx

def generateGraph():
    # 读取稀疏矩阵对象
    from matplotlib import pyplot as plt

    with open('matric/trnMat.pkl', 'rb') as f:
        mat = pickle.load(f)
    # print("mat矩阵：")
    # print(mat)

    # 将稀疏矩阵转换为稠密数组
    mat_array = mat.toarray()

    # 创建 DataFrame 对象
    df = pd.DataFrame(mat_array)

    # 创建一个空的 NetworkX 图对象
    G = nx.Graph()

    # 获取邻接矩阵的行数和列数
    num_rows, num_cols = df.shape
    # print("df矩阵：")
    # print(df.shape)
    #
    # print(num_rows)
    # print(num_cols)
    node1_indices = range(num_rows)
    node2_indices = range(num_cols)

    # 添加第一类节点
    G.add_nodes_from(node1_indices, bipartite=0)

    # 添加第二类节点
    G.add_nodes_from(node2_indices, bipartite=1)

    node_pairs = []

    for i in node1_indices:
        for j in node2_indices:
            if(df.loc[i][j] != 0):
                node_pair = (i,j)
                node_pairs.append(node_pair)
        G.add_edges_from(node_pairs)

    print(nx.info(G))

    node1_indices_list = list(node1_indices)
    node2_indices_list = list(node2_indices)

    return node1_indices_list, node2_indices_list, G

def generateWalks(node1_list, node2_list, graph, steps):
    # 关注度矩阵
    uu_attention_matrix = np.zeros((len(node1_list), len(node1_list)))
    uv_attention_matrix = np.zeros((len(node1_list), len(node2_list)))

    # 遍历所有user节点并获取它们的
    all_walks = []
    for current_node in node1_list:
        walk = [current_node]
        count = 0
        temp_node = current_node
        for _ in range(steps):
            neighbors = list(graph.neighbors(temp_node))
            if len(neighbors) == 0:
                break
            next_node = random.choice(neighbors)

            # 累计user对user节点的经过次数
            if(count % 2 == 0):
                uv_attention_matrix[current_node][temp_node] = uv_attention_matrix[current_node][temp_node] + 1
            else:
            # 累计user对item节点的经过次数
                uu_attention_matrix[current_node][temp_node] = uu_attention_matrix[current_node][temp_node] + 1
            count = count + 1
            walk.append(next_node)
            temp_node = next_node
        all_walks.append(walk)
    current_time = datetime.datetime.now()
    uu_filePath = 'saved_attention_matrix' + '/' + 'UU.npy'
    uv_filePath = 'saved_attention_matrix' + '/' + 'UV.npy'
    np.save(uu_filePath,uu_attention_matrix)
    np.save(uv_filePath,uv_attention_matrix)


if __name__ == '__main__':
    node1_indices_list, node2_indices_list, G = generateGraph()
    generateWalks(node1_indices_list,node2_indices_list,G,80)





# 遍历邻接矩阵的非零元素，添加边
# for i in node1_indices:
#     for j in node2_indices:
#         nonzero_indices = df.iloc[i][df.iloc[j] != 0].index
#         print(nonzero_indices)
#         node_pairs = zip([i] * len(nonzero_indices), num_rows + nonzero_indices)
#         G.add_edges_from(node_pairs)
#
#
# # 打印图的基本信息
# print(nx.info(G))

# def random_walk(graph, start_node, steps):
#     current_node = start_node
#     walk = [current_node]
#
#     for _ in range(steps):
#         neighbors = list(graph.neighbors(current_node))
#         if len(neighbors) == 0:
#             break
#         next_node = random.choice(neighbors)
#         walk.append(next_node)
#         current_node = next_node
#
#     return walk
#
#
# # 定义起始节点和游走步数
# start_node = 0
# steps = 10
#
# # 进行随机游走
# walk = random_walk(G, start_node, steps)
#
# # 打印结果
# print("Random Walk:", walk)