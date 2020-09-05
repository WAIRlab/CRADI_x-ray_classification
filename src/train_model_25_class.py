#============ Basic imports ============#e
import os
import time
import pandas as pd
import gc
import cv2
import csv
import random
from sklearn.metrics.ranking import roc_auc_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from models.models import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
from tuils.loss_function import *
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
# torch.manual_seed(1337)
# torch.cuda.manual_seed(1337)
# np.random.seed(1337)
# random.seed(1337)



def epochVal(model, dataLoader, optimizer, scheduler, loss):

    model.eval ()
    lossVal = 0
    lossValNorm = 0
    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    
    for i, (input, target) in enumerate (dataLoader):

        target = target.view(-1, 25).contiguous().cuda(async=True)
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
    max_threshold, max_result_f1, precision_list, recall_list = search_f1(outPRED, outGT)
    auc = computeAUROC(outGT, outPRED, 25)
    
    return outLoss, auc, max_threshold, max_result_f1, precision_list, recall_list


def train_one_model(model_name):

    snapshot_path = path_data['snapshot_path'] + model_name + '_' + str(Image_size) + '_25_local_val'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    df_all = pd.read_csv(csv_path)
    kfold_path = path_data['k_fold_path']


    for num_fold in range(5):
        print(num_fold)
        # if num_fold in [0,1,2]:
        #     continue

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold]) 
            writer.writerow(['train_batch_size:', str(train_batch_size), 'val_batch_size:', str(val_batch_size), 'backbone', model_name, 'Image_size', Image_size])        
        
        f_train = open(kfold_path + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]

        print('train dataset:', len(c_train), '  val dataset:', len(c_val))
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])  
        # c_train = c_train[0:500]
        # c_val = c_val[0:2000]

        train_loader, val_loader = generate_dataset_loader_25(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        model = DenseNet121_change_avg(25, True)
        model = torch.nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adamax(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        scheduler = WarmRestart(optimizer, T_max=10, T_mult=1, eta_min=1e-5)

        loss = torch.nn.BCELoss(size_average = True)

        trMaxEpoch = 42
        lossMIN = 100000
        val_f1_mean = 0
        val_auc_mean = 0

        for epochID in range (0, trMaxEpoch):

            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 0

            for batchID, (input, target) in enumerate (train_loader):

                target = target.view(-1, 25).contiguous().cuda(async=True)
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)  
                varOutput = model(varInput)
                # print(varOutput.shape, varTarget.shape)
                lossvalue = loss(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1
                optimizer.zero_grad()
                lossvalue.backward()
                optimizer.step()  
            if epochID < 39:
                scheduler.step()
                scheduler = warm_restart(scheduler, T_mult=2)
  
            else:
                optimizer = torch.optim.Adam (model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-08, amsgrad=True)

            trainLoss = trainLoss / lossTrainNorm    
            if (epochID+1)%10 == 0 or epochID > 39 or epochID == 0:

                valLoss, val_auc, val_threshold, val_f1, precision_list, recall_list  = epochVal(model, val_loader, optimizer, scheduler, loss)


            epoch_time = time.time() - start_time
            if valLoss < lossMIN:
                lossMIN = valLoss    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 'val_threshold' : val_threshold, 'val_f1' : val_f1, 'val_f1_mean' : np.mean(val_f1), 'val_auc' : val_auc, 'val_auc_mean' : np.mean(val_auc) }, snapshot_path + '/model_min_loss_' + str(num_fold) +  '.pth.tar')
            if val_f1_mean < np.mean(val_f1):
                val_f1_mean = np.mean(val_f1)    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 'val_threshold' : val_threshold, 'val_f1' : val_f1, 'val_f1_mean' : np.mean(val_f1), 'val_auc' : val_auc, 'val_auc_mean' : np.mean(val_auc) }, snapshot_path + '/model_max_f1_' + str(num_fold) +  '.pth.tar')
            if val_auc_mean < np.mean(val_auc):
                val_auc_mean = np.mean(val_auc)    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 'val_threshold' : val_threshold, 'val_f1' : val_f1, 'val_f1_mean' : np.mean(val_f1), 'val_auc' : val_auc, 'val_auc_mean' : np.mean(val_auc) }, snapshot_path + '/model_max_auc_' + str(num_fold) +  '.pth.tar')
            

            if (epochID+1)%10 == 0:  
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict(), 'val_threshold' : val_threshold, 'val_f1' : val_f1, 'val_f1_mean' : np.mean(val_f1), 'val_auc' : val_auc, 'val_auc_mean' : np.mean(val_auc) }, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth.tar')

            result = [epochID, round(optimizer.state_dict()['param_groups'][0]['lr'], 5), round(trainLoss, 4), round(valLoss, 4), round(epoch_time, 0), round(np.mean(val_f1), 3), round(np.mean(val_auc), 4)]
            print(result)
            # print(val_f1)
            with open(snapshot_path + '/log.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                
                writer.writerow(result + val_threshold + val_f1 + val_auc + precision_list + recall_list)  

        del model    

if __name__ == '__main__':

    import json
    f1 = open('../configs/path_configs_25.json', encoding='utf-8')
    path_data = json.load(f1)
    csv_path = path_data['train_label_path']
    Image_size = 256
    train_batch_size = 48*2
    val_batch_size = 24*2
    workers = 12

    

    backbone_lists = ['DenseNet121_change_avg']
    for backbone in backbone_lists:
        print(backbone)
        train_one_model(backbone)

