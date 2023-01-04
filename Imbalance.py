#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, NearMiss, EditedNearestNeighbours
from sklearn.utils import shuffle

from DataLoader import DataLoader, DataforAgingPrediction

class Imbalance:
    def __init__(self, X, y, sampling="ADASYN", tomek=True):
        self.X = X
        self.y = y
        self.tomek = tomek
        self.IR = 1/1.5
        
        if sampling=="ADASYN":
            self.sampler = ADASYN(sampling_strategy=self.IR, random_state=0)
        
        elif sampling=="BorderlineSMOTE":
            self.sampler = BorderlineSMOTE(sampling_strategy=self.IR, random_state=0, kind="borderline-1")
        
        elif sampling=="SMOTE":
            self.sampler = SMOTE(sampling_strategy=self.IR, random_state=0)
            
        elif sampling=="simple":
            self.sampler = RandomOverSampler(sampling_strategy=self.IR, random_state=0)

        
    def sampling(self):
        print("----------------before sampling----------------")
        label = np.unique(self.y)
        print(f"label {label[0]} : ", len(self.y[self.y==label[0]]))
        print(f"label {label[1]} : ", len(self.y[self.y==label[1]]))

        print("----------------oversampling----------------")
        X_res, y_res = self.sampler.fit_resample(self.X, self.y)
        label = np.unique(y_res)
        print(f"label {label[0]} : ", len(y_res[y_res==label[0]]))
        print(f"label {label[1]} : ", len(y_res[y_res==label[1]]))
        
        print("----------------directly do NearMiss2 undersampling----------------")
        under = NearMiss(sampling_strategy=1/1.5, version=2)
        X_res, y_res = under.fit_resample(X_res, y_res)
        label = np.unique(y_res)
        print(f"label {label[0]} : ", len(y_res[y_res==label[0]]))
        print(f"label {label[1]} : ", len(y_res[y_res==label[1]]))
        
        if self.tomek:
            print("----------------clean sampling data(ENN)----------------")
            X_res, y_res = EditedNearestNeighbours().fit_resample(X_res, y_res)#TomekLinks().fit_resample(X_res, y_res)
            label = np.unique(y_res)
            print(f"label {label[0]} : ", len(y_res[y_res==label[0]]))
            print(f"label {label[1]} : ", len(y_res[y_res==label[1]]))
            
        print("----------------after all of sampling----------------")
        label = np.unique(y_res)
        print(f"label {label[0]} : ", len(y_res[y_res==label[0]]))
        print(f"label {label[1]} : ", len(y_res[y_res==label[1]]))
        
        return X_res, y_res
        
if __name__ == "__main__":
    print("-----------Train Info---------------")
    Train = DataforAgingPrediction("/cccchiang/TabNet/agingdata")
    _traindata = Train.preprocess()
    traindata = Train.clean_traindata(_traindata)
    
    res = Imbalance(traindata.iloc[:,:-1], traindata.iloc[:,-1], sampling="ADASYN")
    X, y = res.sampling()

