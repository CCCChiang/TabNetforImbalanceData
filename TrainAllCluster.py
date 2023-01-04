#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

from Config import Config 
from Tabnet import Tabnet
from Evaluate import Evaluate
from Logger import Logger

class TrainAllCluster():
    def __init__(self, args):
        _path = []
        
        if args["k"]==1:
            
            if args["sampling"]!="No":
                _path.append(f"{args['sampling']}_nocluster")
            
            else:
                _path.append(f"ns_nocluster")
                
        else:
            
            for k in range(args["k"]):
                
                if args["sampling"]!="No":
                    _path.append(f"{args['sampling']}_cluster_{k}")
                
                else:
                    _path.append(f"ns_cluster_{k}")
        
        self._path = _path
        self.args = args
    
    def main(self):
        tn, fp, fn, tp = [], [], [], []
        for k in range(self.args["k"]):
            
            # train
            print(f"--------Training {self.args['root']}/modeldata/{self._path[k]}--------")
            
            tabnet = Tabnet(f"{self.args['root']}/modeldata/{self._path[k]}", 
                            f"{self.args['root']}/pretrain_model/{self._path[k]}",
                            n_d = self.args["n_d"][k], 
                            n_a = self.args["n_a"][k], 
                            n_steps = self.args["n_steps"][k], 
                            gamma = self.args["gamma"][k], 
                            lambda_sparse = self.args["lambda_sparse"][k],
                            max_epochs = self.args["max_epochs"][k], 
                            patience = self.args["patience"][k], 
                            batch_size = self.args["batch_size"][k])
            
            if self.args["IsPreTrain"]:
                print(f"--------Pre-Train--------")
                tabnet.pretrain()
            else:
                print(f"--------No Pre-Train--------")
            
            tabnet.main()
            
            # evaluate
            evaluate = Evaluate(tabnet.val_pred, tabnet.val_y, tabnet.pred, tabnet.test_y)
            threshold_list, val_ar_list, val_recall_list, test_ar_list, test_recall_list, _tn, _fp, _fn, _tp = evaluate.trend_plot(aging_rate_control=self.args["aging_rate_control"], AR_step=0.01, log=False, show_plot=False)
            
            tn.append(_tn)
            fp.append(_fp)
            fn.append(_fn)
            tp.append(_tp)

            # save
            _path = os.path.join(self.args["root"],f"saved_model/{self._path[k]}")
            if os.path.exists(_path):
                pass
            else:
                os.makedirs(_path)
            
            tabnet.clf.save_model(_path)
            
        tn = sum(tn)
        fp = sum(fp)
        fn = sum(fn)
        tp = sum(tp)
        
        # print result
        print("               TRUE")
        print("          ", "OK(0)", "", "NG(1)")
        print("     ", "OK(0)", tn, " | ", fn)
        print(" ", "PRED", "  ", "-------------")
        print("     ", "NG(1)", fp, " | ", tp)

        print("Aging Rate: ", (tp+fp)/(tn+fp+fn+tp))
        print("Recall: ", tp/(fn+tp))
        print("Precision: ", tp/(tp+fp))
        
        logName = "TabNet_IS{}_S{}_IC{}_k{}_scale{}_dm{}_IP{}_ar{}".format(
                    self.args["IsSampling"],
                    self.args["sampling"],
                    self.args["IsClustering"],
                    self.args["k"],
                    self.args["scale"],
                    self.args["pca_dim"],
                    self.args["IsPreTrain"],
                    self.args["aging_rate_control"])
        
        logPath = self.args["root"] + "/log"
        logger = Logger(logName, logPath) ### Declare Log file
        logger.info("---------------Test Data Result--------------")
        logger.info("tn; fn; fp; tp; Aging Rate; Recall; Precision")
        logger.info(f"{round(tn,4)}; {round(fn,4)}; {round(fp,4)}; {round(tp,4)}; {round((tp+fp)/(tn+fp+fn+tp),4)}; {round(tp/(fn+tp),2)}; {round(tp/(tp+fp),4)}")
        
        return tn, fp, fn, tp
        
if __name__ == "__main__":
    root = "/cccchiang/TabNet"
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
#         "Train_path": Train_path,
#         "Test_path": Test_path,
#         "Save_path": Save_path,
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
    
    trainallcluster = TrainAllCluster(args)
    tn, fp, fn, tp = trainallcluster.main()
    