#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DataLoader import DataLoader, DataforAgingPrediction
from TrainTestSplit import TrainTestSplit

class SaveSplitData:
    def __init__(self, train_x, val_x, test_x, train_y, val_y, test_y, path):
        train_x.to_csv(os.path.join(path, "train_x.csv"), index=None)
        val_x.to_csv(os.path.join(path, "val_x.csv"), index=None)
        test_x.to_csv(os.path.join(path, "test_x.csv"), index=None)
        train_y.to_csv(os.path.join(path, "train_y.csv"), index=None)
        val_y.to_csv(os.path.join(path, "val_y.csv"), index=None)
        test_y.to_csv(os.path.join(path, "test_y.csv"), index=None)

if __name__ == "__main__":
    print("-----------Train Info---------------")
    Train = DataforAgingPrediction("/cccchiang/TabNet/***")
    _traindata = Train.preprocess()
    traindata = Train.clean_traindata(_traindata)
    
    print("-----------Test Info---------------")
    Test = DataforAgingPrediction("/cccchiang/TabNet/***/testdata")
    testdata = Test.preprocess()
    _ = Test.clean_traindata(testdata)
    
    print("Align Feature......")
    col_mask = traindata.columns.isin(testdata.columns)
    testdata = testdata.copy()
    for col in traindata.columns[~col_mask]:
        testdata[col] = 0

    testdata = testdata.loc[:,traindata.columns]
    
    datasplit = TrainTestSplit(traindata, testdata, test_size = 0.2)
    train_x, val_x, test_x, train_y, val_y, test_y = datasplit.data_split()
    
    SaveSplitData(train_x, val_x, test_x, train_y, val_y, test_y, "/cccchiang/TabNet/modeldata/ns_nocluster")

