#============ Basic imports ============#
import argparse
import os
import shutil
import time
import tqdm
import glob
from skimage.io import imsave,imread_collection,imread
import pandas as pd
from PIL import Image
import pickle
import gc
import cv2
import csv
import copy
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# cv2.setNumThreads(0)
import matplotlib.pyplot as plt
%matplotlib inline
#============ PyTorch imports ============#
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.nn import Sigmoid
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
import torchvision
import torch.nn.functional as F
import sys  
from tqdm import tqdm
import pydicom
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from collections import OrderedDict

class DenseNet121_change_avg(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_change_avg, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x
    
def forward_pass(model, varInput, index):

    gradients = []
    def save_gradient(grad):
        gradients.append(grad)

    feature_map = model.densenet121(varInput)      
    feature_map = model.relu(feature_map)
    img_feature = model.avgpool(feature_map)
    img_feature = img_feature.view(-1, 1024)
    output = model.mlp(img_feature)
    output = model.sigmoid(output)

    img_feature.register_hook(save_gradient)
    index_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    index_list[index] = 1
    output.backward(torch.FloatTensor([index_list]).cuda())    
    
#     print(output, feature_map.shape, len(gradients))
    return output, feature_map[-1].cpu().data.numpy(), gradients[0].cpu().data.numpy().reshape(-1)   


def create_cam(feature_map, gradients):
    target = feature_map
    weights = gradients
    cam = np.ones(target.shape[1 : ], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * target[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (256, 256))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*(1-mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)/255
    cam = cam / np.max(cam)
    cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
#     plt.imshow(np.uint8(255 * cam))
#     plt.show()
    return np.uint8(255 * cam), np.uint8(255 * heatmap)


def draw_box(image, box, color, thickness=2):
    """ Draws a box on an image with a given color.

    # Arguments
        image     : The image to draw on.
        box       : A list of 4 elements (x1, y1, x2, y2).
        color     : The color of the box.
        thickness : The thickness of the lines to draw a box with.
    """
    b = np.array(box).astype(int)
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)


def draw_caption(image, box, caption):
    """ Draws a caption above the box in an image.

    # Arguments
        image   : The image to draw on.
        box     : A list of 4 elements (x1, y1, x2, y2).
        caption : String containing the text to draw.
    """
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def preprocess_image_imagenet(image):
    """ Preprocess an image: subtracts the pixel mean and divides by the standard deviation.  """

    return (image.astype(np.float32)/255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 加载预训练模型
checkpoint = '/data/VPS/VPS_04/chexnet_pytorch/models_snapshot/DenseNet121_change_avg_256_25/model_min_loss_{fold}.pth.tar'.format(fold=0)
state = torch.load(checkpoint)['state_dict']
model = DenseNet121_change_avg(25).cuda()
# model = nn.DataParallel(model).cuda()  

new_state_dict = OrderedDict()
for k, v in state.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)    
model.eval()

# 读取标签文档
csv_path = '/data/raw_data_repository/chestXray/cleaned_report_25_classes_extraction.csv'
df_all = pd.read_csv(csv_path)

# 选择要预测的图像
c_val = list(set(df_all['Image Index'][df_all['position']=='L\F'][df_all['var_0']==1].values.tolist()))

# 设置显示第几个类的热力图
index = 0
for i in c_val:

    image_path = '/data/raw_data_repository/chestXray/chexpert/local_data/{name}'
    label = torch.FloatTensor(df_all[df_all['Image Index']==i].loc[:, 'var_0':'var_24'].values)

    print(i)
    image = cv2.imread(image_path.format(name=i))
    image = cv2.resize(image, (256,256))
    image_process = preprocess_image_imagenet(image)

    image_tensor = torch.FloatTensor(image_process).unsqueeze(0)
    image_tensor = image_tensor.permute(0,3,1,2).contiguous().float().cuda(async=True)

    varInput = torch.autograd.Variable(image_tensor)

    Output = model(varInput)

    print(label)
    print(Output.data)

    output, feature_map, gradients = forward_pass(model, varInput, index)
    cam = create_cam(feature_map, gradients)
    cam, heatmap = show_cam_on_image(image, cam)

    plt.figure(figsize=(12, 12))
    print(image.shape, cam.shape, heatmap.shape)
    aa = np.hstack((image, cam, heatmap))
    plt.imshow(aa)
    plt.show()
#     cv2.imwrite(image_path.format(name=i.replace('.JPG',  '_heatmap_'+str(index+1)+'.jpg')), cv2.cvtColor(aa, cv2.COLOR_BGR2RGB))
    # plt.savefig('/data/raw_data_repository/chestXray/heatmap_1st_new/'+i.replace('.JPG', '_heatmap_'+str(index+1)+'.svg'))
    # break