#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

from Config import Config 
from DataPipeline import DataPipeline
from TrainAllCluster import TrainAllCluster

def parse_args():
    desc = "~~ TabNet"
    parser = argparse.ArgumentParser(description=desc)

    # data setting
    ## root
    parser.add_argument('--root', type=str, nargs=1, default="/cccchiang/TabNet", help="your root path/ default-> /cccchiang/TabNet")
    
    ## processing
    parser.add_argument('--IsClustering', dest='IsClustering', action='store_true', default=False, help="[True: Clustering/ False: No Clustering/ default-> False]")
    parser.add_argument('--k', type=int, nargs=1, default=1, help="[1: No Clustering/ other int: the numbers of clustering/ default-> 0]")
    
    parser.add_argument('--scale', type=str, nargs=1, default="MA", help="[MA: MaxAbs/ MM: MinMax/ No: Nothing/ default-> MA]")
    parser.add_argument('--pca_dim', type=int, nargs=1, default="0", help="[0: without PCA/ other int: PCA dim/ default-> 0]")
    
    parser.add_argument('--IsSampling', dest='IsSampling', action='store_false', default=True, help="[True: Sampling/ False: No Sampling/ default-> True]")
    parser.add_argument('--sampling', type=str, nargs=1, default="simple", help="[ADASYN/BorderlineSMOTE/SMOTE/simple/No/ default-> simple]")
    parser.add_argument('--tomek', dest='tomek', action='store_false', default=True, help="[True: ENN/ False: No ENN/ default-> True]")

    # model para.
    parser.add_argument('--IsPreTrain', dest='IsPreTrain', action='store_true', default=False, help="[True: PreTrain/ False: No PreTrain/ default-> False]")
    
    parser.add_argument('--aging_rate_control', type=float, nargs=1, default=0.5, help='[aging rate of controling in val. data]')
    parser.add_argument('--n_d', type=int, nargs='+', default=[4,4,4,4], help="可以理解为用来决定输出的隐藏层神经元个数。n_d越大，拟合能力越强，也容易过拟合")
    parser.add_argument('--n_a', type=int, nargs='+', default=[3,3,3,3], help='可以理解为用来决定下一决策步特征选择的隐藏层神经元个数')
    parser.add_argument('--n_steps', type=int, nargs='+', default=[5,5,5,5], help='决策步的个数。可理解为决策树中分裂结点的次数')
    parser.add_argument('--gamma', type=float, nargs='+', default=[1.3,1.3,1.3,1.3], help='决定历史所用特征在当前决策步的特征选择阶段的权重，gamma=1时，表示每个特征在所有决策步中至多仅出现1次')
    parser.add_argument('--lambda_sparse', type=float, nargs='+', default=[1e-3,1e-3,1e-3,1e-3], help='稀疏正则项权重，用来对特征选择阶段的特征稀疏性添加约束,越大则特征选择越稀疏')
    parser.add_argument('--max_epochs', type=int, nargs='+', default=[20,20,20,20], help='最大迭代次数')
    parser.add_argument('--patience', type=int, nargs='+', default=[20,20,20,20], help='在验证集上早停次数')
    parser.add_argument('--batch_size', type=int, nargs='+', default=[256,256,256,256], help='BN作用在的输入特征batch')
    
    return parser.parse_args("")

args = parse_args()
setting = {
    'root': args.root,
    
    "IsClustering": args.IsClustering,
    'k': args.k,
    
    "scale": args.scale,
    "pca_dim": args.pca_dim,
    
    "IsSampling": args.IsSampling,
    'sampling': args.sampling,
    "tomek": args.tomek,
    
    'IsPreTrain': args.IsPreTrain,
    
    'aging_rate_control': args.aging_rate_control,
    'n_d': args.n_d, 
    'n_a': args.n_a, 
    'n_steps': args.n_steps, 
    'gamma': args.gamma,
    'lambda_sparse': args.lambda_sparse,
    'max_epochs': args.max_epochs,
    'patience': args.patience,
    'batch_size': args.batch_size,
}

# config = Config(setting["root"])
# config.save_config(setting)
# args = config.get_config_from_json()

pipeline = DataPipeline(args = setting)
pipeline.main()

trainallcluster = TrainAllCluster(args = setting)
tn, fp, fn, tp = trainallcluster.main()

