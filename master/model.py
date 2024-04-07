import torch
import torch.nn as nn
from utils import sparse_dropout, spmm
import torch.nn.functional as F

class LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, batch_user, device):
        super(LightGCL,self).__init__()
    # 创建一个大小为(n_u, d)的空张量torch.empty(n_u,d)
    # 使用PyTorch中的nn.init.xavier_uniform_()函数对该张量进行Xavier初始化，该函数会根据输入和输出的维度对权重矩阵进行初始化。
    # 将初始化后的张量作为一个可训练参数包装成nn.Parameter对象，并将其赋值给成员变量self.E_u_0
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d))) # 初始化的单个用户嵌入向量 维度为128
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d))) # 初始化的单个物品嵌入向量 维度为128

        self.train_csr = train_csr # 原图的邻接矩阵
        self.adj_norm = adj_norm # 归一化后假图邻接矩阵
        self.l = l # 神经网络层数
        self.E_u_list = [None] * (l+1) # 用户嵌入向量列表 这行代码将创建一个包含l+1个元素的列表self.E_u_list，每个元素的初始值均为None。[None] * (l+1)的操作将会复制None这个元素l+1次，并将这些元素组成一个新的列表
        self.E_i_list = [None] * (l+1) # 物品嵌入向量列表
        self.E_u_list[0] = self.E_u_0 # 将初始化后的 用户嵌入向量加入 user嵌入列表
        self.E_i_list[0] = self.E_i_0 # 将初始化后的 物品嵌入向量加入 item嵌入列表
        self.Z_u_list = [None] * (l+1) # 经过学习后的 用户嵌入向量 列表 对一些边进行了drop操作
        self.Z_i_list = [None] * (l+1) # 经过学习后的 物品嵌入向量 列表 对一些边进行了drop操作
        self.G_u_list = [None] * (l+1) # 假图的嵌入向量
        self.G_i_list = [None] * (l+1) # 假图的嵌入向量
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.batch_user = batch_user

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s # svd后的 u X s
        self.v_mul_s = v_mul_s # svd后的 v(T) X s
        self.ut = ut # u的转置 u(T)
        self.vt = vt # v的转置 v(T)

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False): # 前向传播
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            predictions = preds.argsort(descending=True)
            return predictions
        else:  # training phase
            for layer in range(1,self.l+1): # 循环会执行self.l(2)次
                # GNN propagation

                # 矩阵相乘：经过drop后的归一化邻接矩阵 X 上一层的节点嵌入 当第一次执行时，Z = 邻接矩阵 X 初始化嵌入向量
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1])) # transpose(0, 1)是转置操作

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]  # q.J X J.d -> q.d
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei) # I.q X q.d -> I.d  保存该层的全局用户嵌入到 G_u_list[layer]里
                ut_eu = self.ut @ self.E_u_list[layer-1] # q.I X I.d -> q.d
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu) # J.q X q.d -> J.d 保存该层的全局物品嵌入到 G_i_list[layer]里

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer] # 保存该层的局部用户嵌入到 E_u_list[layer]里
                self.E_i_list[layer] = self.Z_i_list[layer] # 保存该层的局部物品嵌入到 E_i_list[layer]里

            self.G_u = sum(self.G_u_list) # 局部嵌入各层嵌入相加
            self.G_i = sum(self.G_i_list) # 局部嵌入各层嵌入相加

            # aggregate across layers
            self.E_u = sum(self.E_u_list) #
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s
