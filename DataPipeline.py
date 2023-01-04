#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA

from DataLoader import DataLoader, DataforAgingPrediction
from Imbalance import Imbalance
from TrainTestSplit import TrainTestSplit
from Clustering import Clustering
from SaveSplitData import SaveSplitData
from Config import Config 

class DataPipeline:
    def __init__(self, args):
#         self.Train_path = args.Train_path
#         self.Test_path = args.Test_path
#         self.Save_path = args.Save_path
        self.Train_path = args["root"] + "/***"
        self.Test_path = args["root"] + "/***/testdata"
        self.Save_path = args["root"] + "/modeldata"
        self.IsClustering = args["IsClustering"]
        self.IsSampling = args["IsSampling"]
        
        self.k = args["k"]
        self.sampling = args["sampling"]
        self.tomek = args["tomek"]
        self.scale = args["scale"]
        self.pca_dim = args["pca_dim"]

    def main(self):
        
        # data process
        print("-----------Train Info---------------")
        Train = DataforAgingPrediction(self.Train_path)
        _traindata = Train.preprocess()
        traindata = Train.clean_traindata(_traindata)

        print("-----------Test Info---------------")
        Test = DataforAgingPrediction(self.Test_path)
        testdata = Test.preprocess()
        _ = Test.clean_traindata(testdata)
        
        print("Align Feature......")
        col_mask = traindata.columns.isin(testdata.columns)
        testdata = testdata.copy()
        for col in traindata.columns[~col_mask]:
            testdata[col] = 0
        
        testdata = testdata.loc[:,traindata.columns]
        
        # data scaling
        if self.scale == "MM":
            print("MinMax scaling.....")
            scaler = preprocessing.MinMaxScaler()
            traindata.iloc[:,:-1] = scaler.fit_transform(traindata.iloc[:,:-1])
            testdata.iloc[:,:-1] = scaler.transform(testdata.iloc[:,:-1])
        elif self.scale == "Z": 
            print("Standard scaling.....")
            scaler = preprocessing.StandardScaler()
            traindata.iloc[:,:-1] = scaler.fit_transform(traindata.iloc[:,:-1])
            testdata.iloc[:,:-1] = scaler.transform(testdata.iloc[:,:-1])
        elif self.scale == "MA": 
            print("MaxAbs scaling.....")
            scaler = preprocessing.MaxAbsScaler()
            traindata.iloc[:,:-1] = scaler.fit_transform(traindata.iloc[:,:-1])
            testdata.iloc[:,:-1] = scaler.transform(testdata.iloc[:,:-1])
        elif self.scale == "No":
            pass

        # data reduce
        if self.pca_dim==0:
            print("No reduction.....")
            pass
        else:
#             print("PCA reduction.....")
#             pca = PCA(n_components=self.pca_dim)
#             coor = pca.fit_transform(traindata.iloc[:,:-1])
#             traindata = pd.concat([pd.DataFrame(coor),pd.DataFrame(traindata.iloc[:,-1]).reset_index(drop=True)], axis=1)
            
#             coor = pca.transform(testdata.iloc[:,:-1])
#             testdata = pd.concat([pd.DataFrame(coor),pd.DataFrame(testdata.iloc[:,-1]).reset_index(drop=True)], axis=1)
            print("Sparse PCA reduction.....")
            sparse_pca = SparsePCA(n_components=self.pca_dim)
            coor = sparse_pca.fit_transform(traindata.iloc[:,:-1])
            traindata = pd.concat([pd.DataFrame(coor),pd.DataFrame(traindata.iloc[:,-1]).reset_index(drop=True)], axis=1)
            
            coor = sparse_pca.transform(testdata.iloc[:,:-1])
            testdata = pd.concat([pd.DataFrame(coor),pd.DataFrame(testdata.iloc[:,-1]).reset_index(drop=True)], axis=1)
        
        # data clustering
        if self.IsClustering:
            print(f"{self.k} Clustering......")
            clustering = Clustering(traindata = traindata.iloc[:,:-1], testdata = testdata.iloc[:,:-1])
            training_cluster, testing_cluster = clustering.kmeans(k=self.k)
            
            if self.IsSampling:
                print(f"{self.sampling} Sampling......")
#                 res = Imbalance(traindata.iloc[:,:-1], traindata.iloc[:,-1], sampling=self.sampling)
#                 X, y = res.sampling()
#                 traindata = pd.concat([X,y], axis=1)
                
                for i in range(self.k):
                    print(f"-----------Cluster {i}---------------")
                    print("split data...")
                    datasplit = TrainTestSplit(traindata.loc[training_cluster==i,:], testdata.loc[testing_cluster==i,:], test_size = 0.2)
                    train_x, val_x, test_x, train_y, val_y, test_y = datasplit.data_split()
#                     cluster_X = traindata.loc[training_cluster==i,:].iloc[:,:-1]
#                     cluster_y = traindata.loc[training_cluster==i,:].iloc[:,-1]
                    
                    sampling = Imbalance(train_x, train_y, sampling=self.sampling, tomek=self.tomek)
                    cluster_X, cluster_y = sampling.sampling()
#                     cluster_traindata = pd.concat([cluster_X,cluster_y], axis=1)
#                     cluster_testdata = testdata.loc[testing_cluster==i,:]
                    
                    # 
                    _path = os.path.join(self.Save_path,f"{self.sampling}_cluster_{i}")
                    if os.path.exists(_path):
                        pass
                    else:
                        os.mkdir(_path) 
                    
                    print(f"Save the split data to path '{_path}'......")
                    SaveSplitData(cluster_X, val_x, test_x, cluster_y, val_y, test_y, _path)
                    
            else:
                print("No Sampling......")
                for i in range(self.k):
                    print(f"-----------Cluster {i}---------------")
                    cluster_traindata = traindata.loc[training_cluster==i,:]
                    cluster_testdata = testdata.loc[testing_cluster==i,:]
                    
                    # 
                    _path = os.path.join(self.Save_path,f"ns_cluster_{i}")
                    if os.path.exists(_path):
                        pass
                    else:
                        os.mkdir(_path) 
                    
                    print("split data...")
                    print(f"Save the split data to path '{_path}'......")
                    datasplit = TrainTestSplit(cluster_traindata, cluster_testdata, test_size = 0.2)
                    train_x, val_x, test_x, train_y, val_y, test_y = datasplit.data_split()
                    #  save data
                    SaveSplitData(train_x, val_x, test_x, train_y, val_y, test_y, _path)
            
        else:
            print("No Clustering......")
            if self.IsSampling:
                
                print("split data...")
                datasplit = TrainTestSplit(traindata, testdata, test_size = 0.2)
                train_x, val_x, test_x, train_y, val_y, test_y = datasplit.data_split()
                print(f"{self.sampling} Sampling......")
                sampling = Imbalance(train_x, train_y, sampling=self.sampling, tomek=self.tomek)
                X, y = sampling.sampling()
                
                # 
                _path = os.path.join(self.Save_path,f"{self.sampling}_nocluster")
                if os.path.exists(_path):
                    pass
                else:
                    os.mkdir(_path)
                    
                print(f"Save the split data to path '{_path}'......")
                #  save data
                SaveSplitData(X, val_x, test_x, y, val_y, test_y, _path)
                
            else:
                print("No Sampling......")
                # 
                _path = os.path.join(self.Save_path,f"ns_nocluster")
                if os.path.exists(_path):
                    pass
                else:
                    os.mkdir(_path)
                
                print("split data...")
                print(f"Save the split data to path '{_path}'......")
                datasplit = TrainTestSplit(traindata, testdata, test_size = 0.2)
                train_x, val_x, test_x, train_y, val_y, test_y = datasplit.data_split()
                #  save data
                SaveSplitData(train_x, val_x, test_x, train_y, val_y, test_y, _path)
                
        return _path
if __name__ == "__main__":
    root = "/cccchiang/TabNet"
    Train_path = "/cccchiang/TabNet/***"
    Test_path = "/cccchiang/TabNet/***/testdata"
    Save_path = "/cccchiang/TabNet/modeldata"
    
    # control block
    
    ## sampling setting
    IsSampling = True
    tomek = True
    sampling="simple"
    
    ## clustering setting
    IsClustering = False
    k=1
    
    ## scaling setting
    scale = "No"
    pca_dim = 0
    
    ## modeling setting
    IsPreTrain = False    # False  # True
    aging_rate_control = 0.5
    
    # control block
    
    setting = {
        'aging_rate_control': aging_rate_control,
        'IsPreTrain': IsPreTrain,
        'n_d': [4]*k, 
        'n_a': [3]*k, 
        'n_steps': [5]*k, 
        'gamma': [1.3]*k,
        'lambda_sparse': [1e-3]*k,
        'max_epochs': [100]*k,
        'patience': [20]*k,
        'batch_size': [256]*k,
        
        'root': root,
        "Train_path": Train_path,
        "Test_path": Test_path,
        "Save_path": Save_path,
        "scale": scale,
        "pca_dim": pca_dim,
        "IsClustering": IsClustering,
        "IsSampling": IsSampling,
        "tomek": tomek,
        'k': k,
        'sampling': sampling,
    }
    config = Config("/cccchiang/TabNet")
    config.save_config(setting)
    args = config.get_config_from_json()
    
    pipeline = DataPipeline(args)
    pipeline.main()
