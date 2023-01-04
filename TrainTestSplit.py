#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DataLoader import DataLoader, DataforAgingPrediction

class TrainTestSplit():
    def __init__(self, traindata, testdata, test_size = 0.2, random_state=0):
        self.traindata = traindata
        self.testdata = testdata
        self.test_size = test_size
        self.random_state = random_state
        

    def data_split(self):
        OK = self.traindata.loc[self.traindata.GB==0,:]
        NG = self.traindata.loc[self.traindata.GB==1,:]


        #  train/val
        OK_train_x, OK_val_x, OK_train_y, OK_val_y = train_test_split(OK.iloc[:,:(len(OK.columns)-1)], OK.iloc[:, -1], test_size=self.test_size, random_state=self.random_state)
        NG_train_x, NG_val_x, NG_train_y, NG_val_y = train_test_split(NG.iloc[:,:(len(NG.columns)-1)], NG.iloc[:, -1], test_size=self.test_size, random_state=self.random_state)
        train_x, val_x, train_y, val_y = pd.concat([OK_train_x, NG_train_x]), pd.concat([OK_val_x, NG_val_x]), pd.concat([OK_train_y, NG_train_y]), pd.concat([OK_val_y, NG_val_y])

        test_x, test_y = self.testdata.iloc[:,:(len(self.testdata.columns)-1)], self.testdata.iloc[:,-1]

        return train_x, val_x, test_x, train_y, val_y, test_y
    
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

