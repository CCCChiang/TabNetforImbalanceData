#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from matplotlib import pyplot as plt
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer

from DataPipeline import DataPipeline

class Tabnet():
    def __init__(self, data_folder, pre_model_path,
                 n_d = 10, n_a = 10, n_steps = 5, gamma = 1.3, lambda_sparse = 1e-3,
                 max_epochs=10, patience=20, batch_size=128):
        # path
        self.data_folder = data_folder#"/cccchiang/TabNet/modeldata"
        self.pre_model_path = pre_model_path
        
        if os.path.exists(self.pre_model_path):
            pass
        else:
            os.mkdir(self.pre_model_path)
        
        # model parameter
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.max_epochs=max_epochs
        self.patience=patience
        self.batch_size=batch_size
        
        self.tabnet_params = dict(
            n_d = self.n_d,  # 可以理解为用来决定输出的隐藏层神经元个数。n_d越大，拟合能力越强，也容易过拟合
            n_a = self.n_a,   # 可以理解为用来决定下一决策步特征选择的隐藏层神经元个数
            n_steps = self.n_steps, # 决策步的个数。可理解为决策树中分裂结点的次数
            gamma = self.gamma,  # 决定历史所用特征在当前决策步的特征选择阶段的权重，gamma=1时，表示每个特征在所有决策步中至多仅出现1次
            lambda_sparse = self.lambda_sparse,  # 稀疏正则项权重，用来对特征选择阶段的特征稀疏性添加约束,越大则特征选择越稀疏
            optimizer_fn = torch.optim.Adam, # 优化器
            scheduler_params={"step_size":10, # how to use learning rate scheduler
                              "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
            momentum = 0.03,
            mask_type = "entmax",
            seed = 0
        )
        
        # data
        self.train_x = np.array(pd.read_csv(os.path.join(self.data_folder, "train_x.csv")))
        self.val_x = np.array(pd.read_csv(os.path.join(self.data_folder, "val_x.csv")))
        self.test_x = np.array(pd.read_csv(os.path.join(self.data_folder, "test_x.csv")))
        self.train_y = np.array(pd.read_csv(os.path.join(self.data_folder, "train_y.csv"))).ravel()
        self.val_y = np.array(pd.read_csv(os.path.join(self.data_folder, "val_y.csv"))).ravel()
        self.test_y = np.array(pd.read_csv(os.path.join(self.data_folder, "test_y.csv"))).ravel()
#         print(len(self.train_x))
#         print(self.train_x)
#         print(len(self.train_y))
#         print(self.train_y)
#         print(len(self.val_x))
#         print(self.val_x)
#         print(len(self.val_y))
#         print(self.val_y)
#         print(len(self.test_x))
#         print(self.test_x)
#         print(len(self.test_y))
#         print(self.test_y)
        print("train_x shape: ", self.train_x.shape)
        print("train_y shape(0,1): ", (sum(self.train_y==0), sum(self.train_y==1)))
        print("val_x shape: ", self.val_x.shape)
        print("val_y shape(0,1): ", (sum(self.val_y==0), sum(self.val_y==1)))
        print("test_x shape: ", self.test_x.shape)
        print("test_y shape(0,1): ", (sum(self.test_y==0), sum(self.test_y==1)))
        
        # pre-train
        self.pre_train = False
        
    def pretrain(self):
        # TabNetPretrainer
        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax' # "sparsemax"
        )

        unsupervised_model.fit(
            X_train=self.train_x,
            eval_set=[self.val_x],
            batch_size=self.batch_size,
            max_epochs = self.max_epochs,
            virtual_batch_size=128,
            pretraining_ratio=0.8,
        )
        
        unsupervised_model.save_model(os.path.join(self.pre_model_path, "test_pretrain"))
        self.pre_train = True
        
#         reconstructed_X, embedded_X = unsupervised_model.predict(self.val_x)
#         assert(reconstructed_X.shape==embedded_X.shape)

    def main(self):
        self.clf = TabNetClassifier(**self.tabnet_params)
        
        if self.pre_train:
            loaded_pretrain = TabNetPretrainer()
            loaded_pretrain.load_model(os.path.join(self.pre_model_path, "test_pretrain.zip"))
            
            self.clf.fit(
                X_train = self.train_x, y_train=self.train_y,
                eval_set = [(self.train_x, self.train_y), (self.val_x, self.val_y)],
                eval_name = ['train', 'valid'],
                eval_metric = ['auc'],#'logloss', 
                max_epochs = self.max_epochs,  # 最大迭代次数
                patience = self.patience,    # 在验证集上早停次数，
                batch_size = self.batch_size, # BN作用在的输入特征batch
                virtual_batch_size = 16,  # 除了作用于模型输入特征的第一层BN外，都是用的是ghost BN。
                num_workers = 0,
                drop_last = False,
                from_unsupervised=loaded_pretrain
            )
        else:
            self.clf.fit(
                X_train = self.train_x, y_train=self.train_y,
                eval_set = [(self.train_x, self.train_y), (self.val_x, self.val_y)],
                eval_name = ['train', 'valid'],
                eval_metric = ['auc'],#'logloss', 
                max_epochs = self.max_epochs,  # 最大迭代次数
                patience = self.patience,    # 在验证集上早停次数，
                batch_size = self.batch_size, # BN作用在的输入特征batch
                virtual_batch_size = 16,  # 除了作用于模型输入特征的第一层BN外，都是用的是ghost BN。
                num_workers = 0,
                drop_last = False
            )
            
        self.val_pred = self.clf.predict_proba(self.val_x)[:,1]
        self.pred = self.clf.predict_proba(self.test_x)[:,1]
        plt.plot(self.clf.history['loss'])
        self._pred = list(map(lambda x: x>0.5, self.pred))
        print(confusion_matrix(y_true=self.test_y, y_pred=self._pred))
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(y_true=self.test_y, y_pred=self._pred).ravel()
        
        print("Acc: ", accuracy_score(y_pred=self._pred, y_true=self.test_y))
        print("Aging rate: ", (self.tp+self.fp)/(self.tn+self.fp+self.fn+self.tp))
        print("Precision: ", precision_score(y_pred=self._pred, y_true=self.test_y))
        print("recall: ", recall_score(y_pred=self._pred, y_true=self.test_y))
        print("f1: ", f1_score(y_pred=self._pred, y_true=self.test_y))
        print(classification_report(y_pred=self._pred, y_true=self.test_y))

if __name__ == '__main__':
    tabnet = Tabnet("/cccchiang/TabNet/modeldata/ADASYN_cluster_0", 
                    "/cccchiang/TabNet/pretrain_model/ADASYN_cluster_0",
                     n_d = 4, n_a = 3, n_steps = 5, gamma = 1.3, lambda_sparse = 1e-3,
                     max_epochs=5, patience=20, batch_size=256)
#     tabnet.pretrain()
    tabnet.main()

