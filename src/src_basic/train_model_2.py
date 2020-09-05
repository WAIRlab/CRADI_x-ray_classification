#============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn import cross_validation
from sklearn.metrics.ranking import roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# cv2.setNumThreads(0)

#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from models import *
from dataset import *
from tools import *
from lrs_scheduler import WarmRestart, warm_restart, AdamW





torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)



def epochVal(model, dataLoader, optimizer, scheduler, loss):

    model.eval ()

    lossVal = 0
    lossValNorm = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    
    for i, (input, target) in enumerate (dataLoader):

        # input = input.permute(0,3,1,2).contiguous().float().cuda(async=True)
        target = target.view(-1, 2).contiguous().cuda(async=True)
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
    max_threshold, max_result_f2, max_result_me = tp_fp_fn(outPRED[:,0], outGT[:,0].type(torch.cuda.ByteTensor))
    max_threshold_1, max_result_f2_1, max_result_me_1 = tp_fp_fn(outPRED[:,1], outGT[:,1].type(torch.cuda.ByteTensor))
    auc, accuracy = computeAUROC(outGT,outPRED, 2)
    
    return outLoss, auc, accuracy, max_threshold, max_result_f2, max_result_me, max_threshold_1, max_result_f2_1, max_result_me_1 


def train_one_model(model_name):

    snapshot_path = '/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/models_snapshot/' + model_name + '_' + str(Image_size)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    header = ['Epoch', 'Learning rate', 'Train Loss', 'Val Loss', 'Lung Opacity', 'No Lung Opacity / Not Normal', 'Val Acc', 'F2 score', 'Threshold', 'TP/TP+FN+FP', 'F2 score 1', 'Threshold 1', 'TP/TP+FN+FP 1', 'Time']
    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:

            writer = csv.writer(f)
            writer.writerow(['train_batch_size:', str(train_batch_size), 'val_batch_size:', str(val_batch_size), 'backbone', model_name, 'Image_size', Image_size])
            writer.writerow(header)

    df_all = pd.read_csv(csv_path)

    kfold_path = '/data/VPS/VPS_04/pneumonia_detection/pneumonia_ensemble/data/fold_5/'
    for num_fold in range(5):
        print(num_fold)
        # if num_fold != 9:
        #     continue

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold])         
        
        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]

        # c_train = c_train[0:100]
        # c_val = c_val[0:60]

        train_dataset = Kaggle_chest_xray(df_all, c_train, train_transform)
        val_dataset = Kaggle_chest_xray(df_all, c_val, val_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size,        
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,        
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False)
        model = get_model(model_name)
        model = torch.nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adam (model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-08, amsgrad=True)
        # optimizer = AdamW(model.parameters(), lr=1e-3, amsgrad=True)
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

                target = target.view(-1, 2).contiguous().cuda(async=True)
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)  
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
                valLoss, roc_auc, val_acc, max_threshold, max_result_f2, max_result_me, max_threshold_1, max_result_f2_1, max_result_me_1  = epochVal(model, val_loader, optimizer, scheduler, loss)

            epoch_time = time.time() - start_time
            if valLoss < lossMIN:
                lossMIN = valLoss    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'max_result_f2': max_result_f2, 'optimizer' : optimizer.state_dict()}, snapshot_path + '/model_loss_' + str(num_fold) +  '.pth.tar')

            header = ['Epoch', 'Learning rate', 'Train Loss', 'Val Loss', 'Lung Opacity', 'No Lung Opacity / Not Normal', 'Val Acc', 'Time']
            result = [epochID, optimizer.state_dict()['param_groups'][0]['lr'], round(trainLoss, 3), round(valLoss, 3), round(roc_auc[0], 3), round(roc_auc[1], 3), round(val_acc, 3), max_threshold, round(max_result_f2,3), round(max_result_me, 3), max_threshold_1, round(max_result_f2_1,3), round(max_result_me_1, 3), round(epoch_time, 3)]
            print(result)
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)  

        del model    

if __name__ == '__main__':
    csv_path = '/data/raw_data_repository/pneumonia_detection/stage1_image_bbox_full.csv'
    Image_size = 224
    # backbone = 'polynet'

    train_batch_size = 30
    val_batch_size = 15
    workers = 16
    # 'ResNet101','InceptionV4', 'DenseNet121', 'DenseNet161', 'DenseNet169'
    backbone_lists = ['se_resnet152']
    for backbone in backbone_lists:
        train_one_model(backbone)

