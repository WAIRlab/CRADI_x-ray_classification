# -*- coding: utf-8 -*-
import base64
import cv2
import os, io
import numpy as np
import time
import random
import torch
from torch import nn
from torch.nn import functional as F
# import torchvision.models as models
from glob import glob
import json
import logging
# import pretrainedmodels
from collections import OrderedDict

logging.basicConfig(filename='logger.log', level=logging.INFO)

def preprocess_image_imagenet(image):
    """ Preprocess an image: subtracts the pixel mean and divides by the standard deviation.  """

    return (image.astype(np.float32)/255.0 - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)



class BasicConv2d(nn.Module):

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


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
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
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inceptionv4(num_classes=1000, pretrained='imagenet'):

    model = InceptionV4(num_classes=num_classes)
    return model

class InceptionV4_new(nn.Module):

    def __init__(self, classCount):
        super(InceptionV4_new, self).__init__()
        
        # self.model_ft = pretrainedmodels.__dict__['inceptionv4'](num_classes=1001, pretrained='imagenet+background')
        self.model_ft = inceptionv4(num_classes=10, pretrained=None)
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

def load_models():
    # checkpoint_list = glob('/data/VPS/VPS_04/chexnet_api/chexnet_api/uwsgi-flask/app/service/inceptionv3_5k/*.pth.tar')
    # checkpoint_list = glob('/data/VPS/VPS_04/chexnet_api/chexnet_api/uwsgi-flask/app/service/inceptionv3_5k/*_gpu.pth.tar')
    checkpoint_list = glob('inceptionv3_5k/*')
    chexnet_list = []

    for checkpoint in checkpoint_list: 
        model = InceptionV4_new(25)

        new_state_dict = OrderedDict()
        state = torch.load(checkpoint, map_location=lambda storage, loc: storage)['state_dict']
        for k, v in state.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        chexnet_list.append(model)
    return chexnet_list

chexnet_list = load_models()


def inference(gender, chrage, image_id, record_id, raw):

    try:

        if gender not in ['0', '1']:
            logging.error('error_code: 103, message: error gender')
            return {"error_code": 103, "message": "error gender", "data": {}}   
        gender = int(gender)         

        chrage = float(chrage)
        if chrage < 0 or chrage > 100:
            logging.error('error_code: 103, message: error chrage')
            return {"error_code": 103, "message": "error chrage", "data": {}}   

        raw = raw.replace(' ','+')
        if raw == None:
            logging.error('error_code: 103, message: error image')
            return {"error_code": 103, "message": "error image", "data": {}}  

        if image_id == '':
            logging.error('error_code: 103, message: error image_id')
            return {"error_code": 103, "message": "error image_id", "data": {}}  

        if record_id == '':
            logging.error('error_code: 103, message: error id')
            return {"error_code": 103, "message": "error id", "data": {}}  

        try:
            image_data = base64.b64decode(raw)
            im = np.fromstring(image_data,np.uint8)
            im = cv2.imdecode(im,cv2.COLOR_BGR2RGB)
        except Exception as e:
            logging.error('error_code: 103, message: error image')
            return {"error_code": 103, "message": "error image", "data": {}}  

        if im is None:
            logging.error('error_code: 103, message: error image')
            return {"error_code": 103, "message": "error image", "data": {}}

        if len(im.shape) == 2:
            im = cv2.merge([im,im,im])
        if im.shape[2] == 4:
            im = im[:,:,0:3]

        image = cv2.resize(im, (256, 256))
        image = preprocess_image_imagenet(image)
        image_tensor = torch.FloatTensor(image).unsqueeze(0)
        image_tensor = image_tensor.permute(0,3,1,2).contiguous().float()

        with torch.no_grad():
            varInput = torch.autograd.Variable(image_tensor)

        predict = []
        for model in chexnet_list:

            varOutput = model(varInput)[0].detach().cpu().numpy()
            predict.append(varOutput)
        print(predict)
        predict = np.mean(predict, 0).tolist()
        print(predict)
        predict = [round(i, 3) for i in predict]
        logging.info('error_code: 0, message: success, image_id: {image_id}, record_id: {record_id}'.format(image_id=image_id, record_id=record_id))
        return { "error_code": 0, 
                            "message": "success", 
                            "data": {
                                "cls_name":['气胸','肺气肿','肺内钙化','PICC','不张','动脉增宽','动脉弓迂曲','动脉弓钙化','动脉异常','大片影','实变','小片影','心包积液','心影增大','斑片影','肺内阴影','横膈上抬','横膈降低','空洞','纵膈占位','肺内占位','肺淤血','肺纹理增多','肺结节','肺脓肿','肺门影上提','肺门影增大','肺门影模糊','肺门异常','胸腔积液','胸膜增厚','胸膜粘连','胸膜钙化','胸膜异常','脊柱侧弯','起搏器植入后','软组织占位','间质改变'], 
                                "cls_pred":predict, 
                                "heatmap":[],
                                "cls_threshold":[0.36, 0.23, 0.16, 0.15, 0.17, 0.35, 0.33, 0.25, 0.36, 0.18, 0.33, 0.03, 0.27, 0.47, 0.31, 0.26, 0.18, 0.33, 0.43, 0.2, 0.22, 0.45, 0.41, 0.55, 0.32]}}

    except Exception as e:
        print("type error: " + str(e))
        logging.error("type error: " + str(e))


