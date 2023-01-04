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

class Evaluate():
    def __init__(self, val_pred, val_y, test_pred, test_y):
        
        self.threshold = np.unique(val_pred)[np.argsort(np.unique(val_pred))]
        
#         self.val_y = val_y
#         self.val_pred = val_pred
        # np.argsort
        self.val_pred = val_pred[np.argsort(val_pred)]
        self.val_y = val_y[np.argsort(val_pred)]
        self.test_pred = test_pred
        self.test_y = test_y

    
    def pick_threshold(self, aging_rate = 0.5, log=True):
        total_size = len(self.val_pred)
        AR_divid = 2
        

        for i in self.threshold:
            AR = np.sum(self.val_pred>=i)/total_size
            
#             print("AR: ", AR)
            if abs(aging_rate-AR)<=AR_divid:
                AR_divid = abs(aging_rate-AR)
                threshold = i
                ar = 0
                recall = 0
                
            else:
                _val_pred = list(map(lambda x: x>=threshold, self.val_pred))
                tn, fp, fn, tp = confusion_matrix(y_pred=_val_pred, y_true=self.val_y).ravel()
                ar = (tp+fp)/(tn+fp+fn+tp)
                recall = tp/(fn+tp)
                
                if log:
                    print(f"threshold: {threshold} such that aging rate is closest to {aging_rate}")
                    print("--------------------------Val. Metrics---------------------------")
                    print(f"total size: {tn+fp+fn+tp}")
                    print(confusion_matrix(y_pred=_val_pred, y_true=self.val_y))
                    print("Acc: ", accuracy_score(y_pred=_val_pred, y_true=self.val_y))
                    print("Aging rate: ", (tp+fp)/(tn+fp+fn+tp))
                    print("Precision: ", precision_score(y_pred=_val_pred, y_true=self.val_y))
                    print("recall: ", recall_score(y_pred=_val_pred, y_true=self.val_y))
                    print("f1: ", f1_score(y_pred=_val_pred, y_true=self.val_y))
                    print(classification_report(y_pred=_val_pred, y_true=self.val_y))         
                break
                
                
        
        return threshold, ar, recall
            
    def test_metrics(self, threshold, log=True):
        _test_pred = list(map(lambda x: x>=threshold, self.test_pred))
        tn, fp, fn, tp = confusion_matrix( y_pred=_test_pred, y_true=self.test_y).ravel()
        if log:
            print("--------------------------Test. Metrics---------------------------")
            print(f"total size: {tn+fp+fn+tp}")
            print(confusion_matrix(y_pred=_test_pred, y_true=self.test_y))
            print("Acc: ", accuracy_score(y_pred=_test_pred, y_true=self.test_y))
            print("Aging rate: ", (tp+fp)/(tn+fp+fn+tp))
            print("Precision: ", precision_score(y_pred=_test_pred, y_true=self.test_y))
            print("recall: ", recall_score(y_pred=_test_pred, y_true=self.test_y))
            print("f1: ", f1_score(y_pred=_test_pred, y_true=self.test_y))
            print(classification_report(y_pred=_test_pred, y_true=self.test_y))
        else:
            pass
        
        return tn, fp, fn, tp
    
    def trend_plot(self, aging_rate_control=0.5, AR_step=0.1, log=True, show_plot=True):
        threshold_list = []
        val_ar_list = []
        val_recall_list = []
        test_ar_list = []
        test_recall_list = []
        test_precision_list = []
        for i in np.arange(0, 1, AR_step):
            threshold, val_ar, val_recall = self.pick_threshold(i, log)
            tn, fp, fn, tp = self.test_metrics(threshold, log)
            
            test_ar = (tp+fp)/(tn+fp+fn+tp)
            test_recall = tp/(fn+tp)
            test_precision = tp/(tp+fp)
            
            threshold_list.append(threshold)
            val_ar_list.append(val_ar)
            val_recall_list.append(val_recall)
            test_ar_list.append(test_ar)
            test_recall_list.append(test_recall)
            test_precision_list.append(test_precision)
        
        RA_subtract = np.array(list(map(lambda x: abs(x-aging_rate_control), test_ar_list)))
        best_RA_index = np.argsort(RA_subtract)[0]
        
        print("--------------------Result--------------------")
        print("Best Test Aging Rate(most close to 0.5): ", test_ar_list[best_RA_index])
        tn, fp, fn, tp = self.test_metrics(threshold_list[best_RA_index], log)
        print("Test Recall: ", test_recall_list[best_RA_index])
        print("Test Precision: ", test_precision_list[best_RA_index])
        print("Val. Aging Rate: ", val_ar_list[best_RA_index])
        print("Val. Recall: ", val_recall_list[best_RA_index])
        print("Aging Rate threshold: ", np.arange(0, 1, AR_step)[best_RA_index])
        print("Prob. threshold: ", threshold_list[best_RA_index])
        
        if show_plot:
            plt.plot(val_ar_list, val_recall_list, label='Val.')
            plt.plot(test_ar_list, test_recall_list, label='Test')
            plt.vlines(x=0.5, ymin=0, ymax=1, color="red", linestyles="--")
            plt.title("RA-Recall Plot")
            plt.xlabel('Aging Rate')
            plt.ylabel('Recall')
            plt.legend()
            plt.show()

            plt.plot(np.arange(0, 1, AR_step), val_ar_list, label='Val.')
            plt.plot(np.arange(0, 1, AR_step), test_ar_list, label='Test')
            plt.hlines(y=0.5, xmin=0, xmax=1, color="red", linestyles="--")
            plt.title("Aging Rate Plot")
            plt.xlabel('Aging Rate threshold')
            plt.ylabel('Aging Rate')
            plt.legend()
            plt.show()

            plt.plot(np.arange(0, 1, AR_step), val_recall_list, label='Val.')
            plt.plot(np.arange(0, 1, AR_step), test_recall_list, label='Test')
            plt.hlines(y=0.8, xmin=0, xmax=1, color="red", linestyles="--")
            plt.title("Recall Plot")
            plt.xlabel('Aging Rate threshold')
            plt.ylabel('Recall')
            plt.legend()
            plt.show()
            plt.close('all')
        
        return threshold_list, val_ar_list, val_recall_list, test_ar_list, test_recall_list, tn, fp, fn, tp
    
if __name__ == "__main__":
    tabnet = Tabnet("/cccchiang/TabNet/modeldata/ADASYN_cluster_0", 
                    "/cccchiang/TabNet/pretrain_model/ADASYN_cluster_0",
                     n_d = 4, n_a = 3, n_steps = 5, gamma = 1.3, lambda_sparse = 1e-3,
                     max_epochs=2, patience=20, batch_size=256)
#     tabnet.pretrain()
    tabnet.main()
    

    evaluate = Evaluate(tabnet.val_pred, tabnet.val_y, tabnet.pred, tabnet.test_y)
#     threshold, val_ar, val_recall = evaluate.pick_threshold(aging_rate = 0.76)#0.7489211225
#     tn, fp, fn, tp = evaluate.test_metrics(threshold)
    threshold_list, val_ar_list, val_recall_list, test_ar_list, test_recall_list, tn, fp, fn, tp = evaluate.trend_plot(aging_rate_control=0.5, AR_step=0.01, log=False, show_plot=False)

