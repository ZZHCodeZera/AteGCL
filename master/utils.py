import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from scipy import sparse

from parser import args

path = 'data/' + args.data + '/'

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        # 全零数组
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        # ***读取关注度矩阵***
        self.atMat = sparse.load_npz(path+'sparse_matrix_sum.npz').todok()


    def neg_sampling(self):
        # 对每一行
        for i in range(len(self.rows)):
            # u为该行元素列表
            u = self.rows[i]
            while True:
                # 在item编号范围内生成一个随机数
                i_neg = np.random.randint(self.dokmat.shape[1])
                # 如果该行i_neg位置上元素不存在就跳出循环
                if (u, i_neg) not in self.dokmat:
                    break
                # *** 改进点 ***
                # if (u, i_neg) in self.atMat:
                #     break

            # 将该不存在的边添加至负采样数组中
            self.negs[i] = i_neg



    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]