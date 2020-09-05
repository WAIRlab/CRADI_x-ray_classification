import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import math
from sklearn.metrics import fbeta_score
import time
import sys
sys.path.insert(0, '..')  
from src.models import *
from src.dataset import *
from src.tools import *
from src.lrs_scheduler import WarmRestart, warm_restart, AdamW
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

def f2_score(y_true, y_pred):
    # fbeta_score throws a confusing error if inputs are not numpy arrays
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2)

def find_threshold(output, target):
    max_result_f2 = 0
    max_threshold = 0
    beta = 2
    for threshold in [x * 0.01 for x in range(0, 100)]:

        prob = output > threshold
        label = target
        result_f2 = f2_score(label, prob) 
        if result_f2 > max_result_f2:
            max_result_f2 = result_f2
            max_threshold = threshold

    return max_threshold, max_result_f2

def find_threshold_list(output, target):
#     max_result_f2 = 0
#     max_threshold = 0
    beta = 2
    f2_list = []
    max_f2 = []
    max_threshold = []
    for threshold in [x * 0.01 for x in range(0, 100)]:

        prob = output > threshold
        label = target
        result_f2 = f2_score(label, prob) 
        f2_list.append(result_f2)
    
    for i in range(5):
        max_f2.append(f2_list.index(max(f2_list)))
        max_threshold.append(max(f2_list))
        f2_list[f2_list.index(max(f2_list))]=0
    
    return max_threshold, max_f2


def create_data_train(model_lists):
    train_data = pd.DataFrame()

    for model_name in model_lists:
        oof_path = '/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/prediction/{model_name}/val_10.csv'.format(model_name=model_name)
        oof_data = pd.read_csv(oof_path)
        train_data['id'] =  oof_data['id']
        train_data[model_name] = oof_data['prob']
    train_data['class_idx'] = oof_data['class_idx']
    
    return train_data
    
def create_data_test(model_lists):
    val_data = pd.DataFrame()

    for model_name in model_lists:
        oof_path = '/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/prediction/{model_name}/test_10.csv'.format(model_name=model_name)
        oof_data = pd.read_csv(oof_path)
        val_data['id'] =  oof_data['id']
        val_data[model_name] = oof_data['prob']
    
    return val_data

model_lists = ['InceptionV4', 'ResNet18', 'ResNet50', 'ResNet101', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 'se_resnet101', 'se_resnet152']

train_data = create_data_train(model_lists)
test_data   = create_data_test(model_lists)

label = np.array(train_data['class_idx'])
label[label>0.5] = 1
label = 1 - label

train = np.array(train_data[model_lists])
test = np.array(test_data[model_lists])


import time

def xgboost_random_step(random_state, iter, train, label, test):
    start_time = time.time()
    rs = random_state + iter

    num_folds = random.randint(4, 10)
    eta = random.uniform(0.06, 0.45)
    max_depth = random.randint(2, 6)
    subsample = random.uniform(0.6, 0.99)
    colsample_bytree = random.uniform(0.6, 0.99)
    # eval_metric = random.choice(['auc', 'logloss'])
    eval_metric = 'logloss'
    # eta = 0.1
    # max_depth = 3
    # subsample = 0.95
    # colsample_bytree = 0.95
    log_str = 'XGBoost iter {}. FOLDS: {} METRIC: {} ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(iter,
                                                                                                           num_folds,
                                                                                                           eval_metric,
                                                                                                           eta,
                                                                                                           max_depth,
                                                                                                           subsample,
                                                                                                           colsample_bytree)
    print(log_str)
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": eval_metric,
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": rs,
        "nthread": 6,
        # 'gpu_id': 0,
        # 'updater': 'grow_gpu_hist',
    }
    num_boost_round = 1000
    early_stopping_rounds = 40

    validation_arr = np.zeros(label.shape)
#     validation_arr[:, :] = -1

    test_preds = np.zeros((num_folds, test.shape[0]))

    kf = KFold(train.shape[0], shuffle=True, n_folds=num_folds, random_state=rs)

    for num_fold, (train_index, test_index) in enumerate(kf):
        train_X, valid_X = train[train_index], train[test_index]
        train_y, valid_y = label[train_index], label[test_index]
#         print(train_X.shape, train_y.shape)
    #     print(train_X, train_y)

        d_train = xgb.DMatrix(train_X, train_y)
        d_valid = xgb.DMatrix(valid_X, valid_y)


        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        

        gbm = xgb.train(params, d_train, num_boost_round, evals=watchlist,
                        early_stopping_rounds=early_stopping_rounds, verbose_eval=False)

        # print("Validating...")
        preds = gbm.predict(d_valid, ntree_limit=gbm.best_iteration + 1)
        # print(valid_y, preds, valid_y.shape, preds.shape)
        bce_loss = log_loss(valid_y, preds)

        max_threshold, max_result_f2 = find_threshold_list(preds, valid_y)
#         print(bce_loss, max_threshold, max_result_f2)

        d_test = xgb.DMatrix(test)
        test_preds[num_fold, :] = gbm.predict(d_test, ntree_limit=gbm.best_iteration + 1)
        validation_arr[test_index] = preds


    print("Time XGBoost: %s sec" % (round(time.time() - start_time, 0)))
    return validation_arr, test_preds, log_str, params

for i in range(1000):
    random_state = random.randint(0,8888)
    random.seed(random_state)
    validation_arr, test_preds, log_str, params = xgboost_random_step(random_state, i, train, label, test)

    oof_loss = log_loss(label, validation_arr)
    oof_max_result_f2, oof_max_threshold = find_threshold_list(validation_arr, label)
    print(oof_loss, oof_max_threshold, oof_max_result_f2)

    if oof_max_result_f2[0] > 0.755:
        test_preds = test_preds.mean(0)
        test_vote = test_preds.copy()
        test_vote[test_vote > oof_max_threshold[0]*0.01] = 1
        test_vote[test_vote < oof_max_threshold[0]*0.01] = 0

        test_data['pred'] = test_preds
        test_data['vote'] = test_vote
        test_data.to_csv('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/xgboost/xgboost_ensemble_test_{iter}_{oof_loss}_{oof_max_threshold}_{oof_max_result_f2}.csv'.format(iter=i,oof_loss=oof_loss,oof_max_threshold=oof_max_threshold[0],oof_max_result_f2=oof_max_result_f2[0]), index=0)
        

        train_data['prob'] = validation_arr
        train_data.to_csv('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/xgboost/xgboost_ensemble_val_{iter}_{oof_loss}_{oof_max_threshold}_{oof_max_result_f2}.csv'.format(iter=i,oof_loss=oof_loss,oof_max_threshold=oof_max_threshold[0],oof_max_result_f2=oof_max_result_f2[0]), index=0)
