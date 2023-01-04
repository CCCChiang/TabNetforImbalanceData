#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from multiprocessing import cpu_count
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from DataLoader import DataLoader, DataforAgingPrediction

class Clustering:
    def __init__(self, traindata, testdata):
        self.traindata = traindata
        self.testdata = testdata

    def kmeans(self, k=4):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(self.traindata)
        training_cluster = kmeans.labels_
        testing_cluster = kmeans.predict(self.testdata)
        
        
        return training_cluster, testing_cluster
    
    def elbow(self, max_k=11, paralize = True):
        
        def kmeansRes(k=4):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(self.traindata)
            distortions = kmeans.inertia_

            return distortions
        
        if paralize:
            pool_sz = cpu_count()
            res = Parallel(n_jobs=pool_sz)(delayed(kmeansRes)(k) for k in range(2, max_k))
            
        else:
            res = []
            for i in range(2, max_k):
                res.append(kmeansRes(k=i))
        
        return res

    

