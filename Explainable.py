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
from sklearn.metrics import roc_auc_score

from Tabnet import Tabnet

class Explainable:
    def __init__(self, data_folder, model):
        self.data_folder = data_folder
        self.model = model
        self.traindata = pd.read_csv(os.path.join(self.data_folder, "train_x.csv"))
        self.data_col = self.traindata.columns
    
    def cal_importance(self):
        # for TabNet
        sort_index = np.argsort(self.model.clf.feature_importances_)
        
        # importance/important_feature
        importance = self.model.clf.feature_importances_[sort_index][::-1]
        important_feature = self.data_col[sort_index][::-1]
        
        # for plot
        parameters = {'ytick.labelsize': 5}
        plt.rcParams.update(parameters)
        plt.barh(important_feature[sort_index], importance)
        plt.show()
        
        return importance, important_feature
    
    def cal_mask(self):
        # for testdata
        explain_matrix, masks = self.model.clf.explain(self.model.test_x)
        fig, axs = plt.subplots(1, self.model.clf.n_steps, figsize=(20,20))

        for i in range(self.model.clf.n_steps):
            axs[i].imshow(masks[i][:50])
            axs[i].set_title(f"mask {i}")
        plt.show()

        return explain_matrix, masks

if __name__ == "__main__":
    tabnet = Tabnet("/cccchiang/TabNet/modeldata/SMOTE_nocluster", 
                     n_d = 10, n_a = 10, n_steps = 5, gamma = 1.3, lambda_sparse = 1e-3,
                     max_epochs=10, patience=20, batch_size=128)
    tabnet.main()
    
    explain = Explainable(data_folder = "/cccchiang/TabNet/modeldata/SMOTE_nocluster", model = tabnet)
    importance, important_feature = explain.cal_importance()
    explain_matrix, masks = explain.cal_mask()

