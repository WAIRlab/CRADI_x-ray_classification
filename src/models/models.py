import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
# from models.densenet_efficient import *
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels
import math
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import *
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.unet.decoder import UnetDecoder

class diy_model(nn.Module):

    def __init__(self):
        super(diy_model, self).__init__()
        self.model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.cls_head = nn.Sequential(nn.Linear(2048, 2048, bias=True), nn.Linear(2048, 1, bias=True))
        self.fea_bn = nn.BatchNorm1d(512)
        self.fea_bn.bias.requires_grad_(False)

    def forward(self, x):
        
        global_features = self.model.encoder(x)
        
        cls_feature = global_features[0]
        cls_feature = self.avgpool(cls_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        
        # cls_feature = self.fea_bn(cls_feature)
        cls_feature = self.cls_head(cls_feature)
        
        seg_feature = self.model.decoder(global_features)
        return cls_feature, seg_feature



class cbam_resnet50_osmr(nn.Module):

    def __init__(self, classCount):
        super(cbam_resnet50_osmr, self).__init__()

        self.model_ft = ptcv_get_model("cbam_resnet50", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x
        
class se_resnet101(nn.Module):

    def __init__(self, classCount):
        super(se_resnet101, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

        
class ibn_resnext101_32x4d_osmr(nn.Module):

    def __init__(self, classCount):
        super(ibn_resnext101_32x4d_osmr, self).__init__()

        self.model_ft = ptcv_get_model("ibn_resnext101_32x4d", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x
        
class airnext50_32x4d_r2_osmr(nn.Module):

    def __init__(self, classCount):
        super(airnext50_32x4d_r2_osmr, self).__init__()

        self.model_ft = ptcv_get_model("airnext50_32x4d_r2", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class Xception_osmr(nn.Module):

    def __init__(self, classCount):
        super(Xception_osmr, self).__init__()

        self.model_ft = ptcv_get_model("xception", pretrained=True)
        num_ftrs = self.model_ft.output.in_features
        self.model_ft.features.final_block.pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.output = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class inceptionresnetv2(nn.Module):

    def __init__(self, classCount):
        super(inceptionresnetv2, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext101_32x4d(nn.Module):

    def __init__(self, classCount):
        super(se_resnext101_32x4d, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class se_resnext50_32x4d(nn.Module):

    def __init__(self, classCount):
        super(se_resnext50_32x4d, self).__init__()

        self.model_ft = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount, bias=True),
                                             nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class DenseNet201_change_avg(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet201_change_avg, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1920, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet201(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x

class DenseNet169_change_avg(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet169_change_avg, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained).features
        # self.densenet121[0] = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
        # #                                     # nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # #                                     # nn.ReLU(inplace=True),
        #                                     self.densenet121[0])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=65, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        # print('inputs:', inputs.shape)
        cos_th = F.linear(inputs, F.normalize(self.weight))
        # print('cos_th:', cos_th.shape)
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = labels
        # print(labels.shape, labels)
        # onehot = torch.zeros(cos_th.size()).cuda()
        # print('onehot:', cos_th.shape)
        # onehot.scatter_(1, labels, 1)
        # print(onehot.shape, onehot)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

class DenseNet121_change_avg_arcface_loss(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_change_avg_arcface_loss, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1024*8*8, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.arc = ArcModule(512, 28, s=65, m=0.1)
        self.bn = nn.BatchNorm2d(1024)
        self.dropout = nn.Dropout2d(0.75, inplace=True)
        

    def forward(self, x, labels):
        # labels = labels.long().cuda()
        features = self.densenet121(x)      
        features = self.bn(features)
        features = self.dropout(features)
        # features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn1(features)
        features = F.normalize(features)
        arcface = self.arc(features, labels)
        out = torch.sigmoid(arcface)
        # print(arcface.shape, arcface)
        return out


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


class InceptionV4(nn.Module):

    def __init__(self, classCount):
        super(InceptionV4, self).__init__()
        
        self.model_ft = pretrainedmodels.__dict__['inceptionv4'](num_classes=1001, pretrained='imagenet+background')
        # self.model_ft.features[0].conv = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
        #                                                nn.BatchNorm2d(3, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
        #                                                nn.ReLU(inplace),
        #                                                self.model_ft.features[0].conv)
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class nasnetamobile(nn.Module):

    def __init__(self, classCount):
        super(nasnetamobile, self).__init__()
        
        self.model_ft = pretrainedmodels.__dict__['nasnetamobile'](num_classes=1000, pretrained='imagenet')
        # self.model_ft.features[0].conv = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False),
        #                                                nn.BatchNorm2d(3, eps=0.001, momentum=0.1, affine=True, track_running_stats=True),
        #                                                nn.ReLU(inplace),
        #                                                self.model_ft.features[0].conv)
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.model_ft(x)
        return x

class ResNet101(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101()
        self.resnet101.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet101.fc = nn.Sequential(nn.Linear(2048, classCount),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)      
        return x

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet50, self).__init__()
        self.resnet50 = torchvision.models.resnet50()
        self.resnet50.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet50.fc = nn.Sequential(nn.Linear(2048, classCount),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)      
        return x

def get_model(backbone):

    if backbone == 'DenseNet121_change_avg':
        model = DenseNet121_change_avg(10, True)
    elif backbone == 'InceptionV4':
        model = InceptionV4(25)
    elif backbone == 'DenseNet169_change_avg':
        model = DenseNet169_change_avg(23, True)
    elif backbone == 'se_resnext101_32x4d':
        model = se_resnext101_32x4d(25)
    elif backbone == 'nasnetamobile':
        model = nasnetamobile(23)
    elif backbone == 'DenseNet201_change_avg':
        model = DenseNet201_change_avg(23, True)
    elif backbone == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(25)
    elif backbone == 'Xception_osmr':
        model = Xception_osmr(23)
    elif backbone == 'ibn_resnext101_32x4d_osmr':
        model = ibn_resnext101_32x4d_osmr(23)
    elif backbone == 'se_resnet101':
        model = se_resnet101(23)
    elif backbone == 'DenseNet121_change_avg_arcface_loss':
        model = DenseNet121_change_avg_arcface_loss(23)  
   
    return model