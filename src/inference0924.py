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
#matplotlib inline
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
#matplotlib inline
import numpy as np
from collections import OrderedDict

class BasicConv2d(nn.Module):##定义了一个卷积层+bn+relu

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):##maxpool+conv+bn+relu

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)##拼接
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
 
        self.last_linear = nn.Linear(1536, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        self.image2 = features
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
#        self.image2=x
        x = x.view(x.size(0), -1)
        self.image=x
        x = self.last_linear(x)
        return x, self.image, self.image2

    def forward(self, input):
        x = self.features(input)
        x, image, image2 = self.logits(x)
        return x, image,image2


def inceptionv4(num_classes=1000, pretrained='imagenet'):

    model= InceptionV4(num_classes=num_classes)
    return model

class InceptionV4_new(nn.Module):

    def __init__(self, classCount):
        super(InceptionV4_new, self).__init__()
        
        # self.model_ft = pretrainedmodels.__dict__['inceptionv4'](num_classes=1001, pretrained='imagenet+background')
        self.model_ft= inceptionv4(num_classes=classCount, pretrained=None)
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount), nn.Sigmoid())  

    def forward(self, x):
        x,y,z = self.model_ft(x)
        return x, y, z

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
    a,b,c=model(varInput)

    feature_map = b

    #R=nn.ReLU() 
    #feature_map = R(feature_map)
    #mlp = nn.Linear(1536, 25)
    #sigmoid = nn.Sigmoid()
    #avgpool = nn.AdaptiveAvgPool2d((1,1))
    #img_feature =avgpool(feature_map)
    #img_feature = img_feature.view(-1, 1536)
    #output = mlp(img_feature)
    #output = sigmoid(output)


    img_feature = feature_map.view(-1, 1536)
    img_feature.register_hook(save_gradient)
    index_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    index_list[index] = 1
    L=nn.Linear(1536, 25)
    output=L(img_feature)
    S=nn.Sigmoid()
    output=S(output)
    output.backward(torch.FloatTensor([index_list]))    
    print('gradients', end=' ')
    print(gradients)
    #print(output, feature_map.shape, len(gradients))
    return output, c[-1].cpu().data.numpy(), gradients[0].cpu().data.numpy().reshape(-1)   


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
    print(feature_map.shape)
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
checkpoint = './model_min_loss_{fold}.pth.tar'.format(fold=0)
state = torch.load(checkpoint,map_location='cpu')['state_dict']
#model = DenseNet121_change_avg(25).cuda()
model = InceptionV4_new(25)
# model = nn.DataParallel(model).cuda()  
new_state_dict = OrderedDict()
for k, v in state.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict) 
model.eval()

# 读取标签文档
csv_path = '/CRADI_label_sample.xlsx'
df_all = pd.read_csv(csv_path)

# 选择要预测的图像
#c_val = list(set(df_all['Image Index'][df_all['position']=='L\F'][df_all['var_0']==1].values.tolist()))
c_val=['image00000004.png','image00000005.png']
# 设置显示第几个类的热力图
index = 2
for i in c_val:

    image_path = 'CRADI_x-ray_classification/images/{name}'
    label = torch.FloatTensor(df_all[df_all['new_id']==i[:13]].loc[:, 'PICC implant':'small consolidation'].values)

    print(i)
    image = cv2.imread(image_path.format(name=i))
    image = cv2.resize(image, (256,256))
    image_process = preprocess_image_imagenet(image)

    image_tensor = torch.FloatTensor(image_process).unsqueeze(0)
#    image_tensor = image_tensor.permute(0,3,1,2).contiguous().float().cuda(async=True)
    image_tensor = image_tensor.permute(0,3,1,2).contiguous().float()

    varInput = torch.autograd.Variable(image_tensor)

    Output = model(varInput)

    print(label)
#    print(Output.data)

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