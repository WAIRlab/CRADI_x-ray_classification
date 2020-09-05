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



import torchvision.datasets as datasets
import torch.utils.data as data
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Mlp_1(torch.nn.Module):
    def __init__(self, num_of_res, num_of_inp,):
        super(Mlp_1, self).__init__()

        
        self.layers = []
        self.layers.append(nn.Linear(num_of_inp, 500))
        self.layers.append(nn.BatchNorm1d(500))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.3))
        self.layers.append(nn.Linear(500, 300))
        self.layers.append(nn.BatchNorm1d(300))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.3))
        self.layers.append(nn.Linear(300, 1))
        self.layers.append(nn.Sigmoid())


        self.net = nn.Sequential(*self.layers)

    def forward(self, input):

        out = self.net(input)
        return out
    
class Feature_Dataset_train(data.Dataset):
    def __init__(self, train, label):
        self.train = train
        self.label = label

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        X = self.train[idx]
        y = self.label[idx]
        return X, y

class Feature_Dataset_test(data.Dataset):
    def __init__(self, test):
        self.test = test

    def __len__(self):
        return self.test.shape[0]

    def __getitem__(self, idx):
        X = self.test[idx]
        return X

def epochVal(model, dataLoader, optimizer, scheduler, loss):

    model.eval ()

    lossVal = 0
    lossValNorm = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    
    for i, (input, target) in enumerate (dataLoader):

        input = input.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.FloatTensor)
            
        target = target.view(-1, 1).contiguous().cuda(async=True)
        outGT = torch.cat((outGT, target), 0)
        with torch.no_grad():
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)    
        varOutput = model(varInput)

        losstensor = loss(varOutput, varTarget)
        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossVal += losstensor.item()
        lossValNorm += 1
    
    outLoss = lossVal / lossValNorm
    auc, accuracy = computeAUROC(outGT,outPRED, 1)
    outGT = outGT.cpu().numpy()
    outPRED = outPRED.cpu().numpy()
    
    return outLoss, auc, outGT, outPRED


def epochTest(model, dataLoader):

    model.eval ()
    outPRED = torch.FloatTensor().cuda()
    
    for i, (input) in enumerate (dataLoader):

        input = input.type(torch.cuda.FloatTensor)
            
        with torch.no_grad():
            varInput = torch.autograd.Variable(input)  
        varOutput = model(varInput)
        outPRED = torch.cat((outPRED, varOutput.data), 0)

    outPRED = outPRED.cpu().numpy()
    
    return outPRED


class Mlp(torch.nn.Module):
    def __init__(self, num_of_res, num_of_inp, params):
        super(Mlp, self).__init__()

        num_layer = params['num_layer']
        level1_size = 300
        if 'level1_size' in params:
            level1_size = params['level1_size']
        level2_size = 300
        if 'level2_size' in params:
            level2_size = params['level2_size']
        level3_size = 250
        if 'level3_size' in params:
            level3_size = params['level3_size']

        dropout_val_1 = 0.1
        if 'dropout_val_1' in params:
            dropout_val_1 = params['dropout_val_1']
        dropout_val_2 = 0.1
        if 'dropout_val_2' in params:
            dropout_val_2 = params['dropout_val_2']
        dropout_val_3 = 0.1
        if 'dropout_val_3' in params:
            dropout_val_3 = params['dropout_val_3']

        activation_1 = 'prelu'
        if 'activation_1' in params:
            activation_1 = params['activation_1']
        activation_2 = 'prelu'
        if 'activation_2' in params:
            activation_2 = params['activation_2']
        activation_3 = 'prelu'
        if 'activation_3' in params:
            activation_3 = params['activation_3']

        use_3rd_level = 1
        if 'use_3rd_level' in params:
            use_3rd_level = params['use_3rd_level']
        
        self.layers = []

        self.layers.append(nn.Linear(num_of_inp, level1_size))
        self.layers.append(nn.BatchNorm1d(level1_size))
        self.layers.append(nn.Dropout(dropout_val_1))
        if activation_1 == 'prelu':
            self.layers.append(nn.PReLU())
        elif activation_1 == 'relu':
            self.layers.append(nn.PReLU())
        elif activation_1 == 'leakyrelu':
            self.layers.append(nn.LeakyReLU())
            
        if num_layer in [2, 3]:            
            self.layers.append(nn.Linear(level1_size, level2_size))
            self.layers.append(nn.BatchNorm1d(level2_size))
            self.layers.append(nn.Dropout(dropout_val_2))
            if activation_2 == 'prelu':
                self.layers.append(nn.PReLU())
            elif activation_2 == 'relu':
                self.layers.append(nn.PReLU())
            elif activation_2 == 'leakyrelu':
                self.layers.append(nn.LeakyReLU())

        if num_layer in [3]:
            self.layers.append(nn.Linear(level2_size, level3_size))
            self.layers.append(nn.BatchNorm1d(level3_size))
            self.layers.append(nn.Dropout(dropout_val_3))
            if activation_3 == 'prelu':
                self.layers.append(nn.PReLU())
            elif activation_3 == 'relu':
                self.layers.append(nn.PReLU())
            elif activation_3 == 'leakyrelu':
                self.layers.append(nn.LeakyReLU())
        
        if num_layer == 1:
            self.layers.append(nn.Linear(level1_size, 1))
        elif num_layer == 2:
            self.layers.append(nn.Linear(level2_size, 1))            
        elif num_layer == 3:
            self.layers.append(nn.Linear(level3_size, 1))
            
        self.layers.append(nn.Sigmoid())


        self.net = nn.Sequential(*self.layers)

    def forward(self, input):

        out = self.net(input)
        return out






def random_keras_step(random_state, iter, train, label, test):

    start_time = time.time()
    rs = random_state + iter

    num_folds = random.randint(4, 10)
    batch_size = random.randint(200, 500)

    mlp_param = dict()
    
    mlp_param['num_layer'] = random.randint(1, 3)
    
    
    mlp_param['dropout_val_1'] = random.uniform(0.05, 0.5)
    mlp_param['dropout_val_2'] = random.uniform(0.1, 0.5)
    mlp_param['dropout_val_3'] = random.uniform(0.1, 0.5)
    mlp_param['level1_size'] = random.randint(400, 700)
    mlp_param['level2_size'] = random.randint(350, 600)
    mlp_param['level3_size'] = random.randint(200, 500)
    mlp_param['activation_1'] = random.choice(['prelu', 'relu', 'leakyrelu'])
    mlp_param['activation_2'] = random.choice(['prelu', 'relu', 'leakyrelu'])
    mlp_param['activation_3'] = random.choice(['prelu', 'relu', 'leakyrelu'])



    log_str = 'Keras iter {}. FOLDS: {} BATCH: {} LAYER: {}'.format(
        iter,
        num_folds,
        batch_size,
        mlp_param['num_layer'])
    print(log_str)
    print('CNN params: {}'.format(mlp_param))

    validation_arr = np.zeros(label.shape)
    test_preds = np.zeros((5, test.shape[0]))

    kf = KFold(train.shape[0], shuffle=True, n_folds=5, random_state=6)

    for num_fold, (train_index, test_index) in enumerate(kf):
        X_train, X_valid = train[train_index], train[test_index]
        y_train, y_valid = label[train_index], label[test_index]

        train_dataset = Feature_Dataset_train(X_train, y_train)
        val_dataset = Feature_Dataset_train(X_valid, y_valid)
        test_dataset = Feature_Dataset_test(test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,        
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,        
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False)
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,        
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            drop_last=False)
        
        model = Mlp(1, X_train.shape[1], mlp_param)
#         model = Mlp_1(1, X_train.shape[1])
        model = torch.nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adam (model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-08, amsgrad=True)
        scheduler = WarmRestart(optimizer, T_max=10, T_mult=1, eta_min=1e-5)

        loss = torch.nn.BCELoss(size_average = True)

        trMaxEpoch = 13
        lossMIN = 100000
        f2MAX = 0

        for epochID in range (0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 0

            for batchID, (input, target) in enumerate (train_loader):

                target = target.view(-1, 1).contiguous().cuda(async=True)
                input = input.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.FloatTensor)
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)  
    #             print(varInput, varInput.size())
                varOutput = model(varInput)
                lossvalue = loss(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1

                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()  
            if epochID < 10:
                # for warm_restart
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
                optimizer.step()
            else:
                optimizer = torch.optim.Adam (model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=True)

            trainLoss = trainLoss / lossTrainNorm    
            if epochID%1 == 0:
                valLoss, roc_auc, val_label, preds  = epochVal(model, val_loader, optimizer, scheduler, loss)
            
            if valLoss < lossMIN:
                lossMIN = valLoss    
                val_lable_max, preds_max = val_label, preds
            epoch_time = time.time() - start_time
            result = [epochID, optimizer.state_dict()['param_groups'][0]['lr'], round(trainLoss, 3), round(valLoss, 3), round(roc_auc[0], 3), round(epoch_time, 3)]
#             print(result)
        bce_loss = log_loss(val_lable_max.ravel(), preds_max.ravel())

        max_threshold, max_result_f2 = find_threshold_list(preds_max.ravel(), val_lable_max.ravel())
        print(bce_loss, max_threshold, max_result_f2)
        validation_arr[test_index] = preds.ravel()
        test_preds[num_fold,:] = epochTest(model, test_loader).ravel()

#     print("Time Keras: %s sec" % (round(time.time() - start_time, 0)))
    return validation_arr, test_preds



for i in range(500):
    random_state = random.randint(0,8888)
    random.seed(random_state)
    validation_arr, test_preds = random_keras_step(random_state, i, train, label, test)

    oof_loss = log_loss(label, validation_arr)
    oof_max_result_f2, oof_max_threshold = find_threshold_list(validation_arr, label)
    print(oof_loss, oof_max_threshold, oof_max_result_f2)
    
    if oof_max_result_f2[0] > 0.75:
        test_preds = test_preds.mean(0)
    #     print(test_preds.shape)
        test_vote = test_preds.copy()
        test_vote[test_vote > oof_max_threshold[0]*0.01] = 1
        test_vote[test_vote < oof_max_threshold[0]*0.01] = 0

        test_data['pred'] = test_preds
        test_data['vote'] = test_vote
        test_data.to_csv('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/mlp/mlp_ensemble_test_{iter}_{oof_loss}_{oof_max_threshold}_{oof_max_result_f2}.csv'.format(iter=i,oof_loss=oof_loss,oof_max_threshold=oof_max_threshold[0],oof_max_result_f2=oof_max_result_f2[0]), index=0)
        train_data['prob'] = validation_arr
        train_data.to_csv('/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/mlp/mlp_ensemble_val_{iter}_{oof_loss}_{oof_max_threshold}_{oof_max_result_f2}.csv'.format(iter=i,oof_loss=oof_loss,oof_max_threshold=oof_max_threshold[0],oof_max_result_f2=oof_max_result_f2[0]), index=0)
