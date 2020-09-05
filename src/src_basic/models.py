import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch
import pretrainedmodels

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        self.densenet169 = torchvision.models.densenet169(pretrained=isTrained)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        self.densenet201 = torchvision.models.densenet201(pretrained=isTrained)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x

class DenseNet161(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet161, self).__init__()
        
        self.densenet161 = torchvision.models.densenet161(pretrained=isTrained)
        
        kernelCount = self.densenet161.classifier.in_features
        
        self.densenet161.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet161(x)
        return x

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(ResNet18, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)

        kernelCount = self.resnet18.fc.in_features
        
        self.resnet18.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet34(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(ResNet34, self).__init__()
        
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)

        kernelCount = self.resnet34.fc.in_features
        
        self.resnet34.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet34(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(ResNet50, self).__init__()
        
        self.resnet50 = torchvision.models.resnet50(pretrained=isTrained)

        kernelCount = self.resnet50.fc.in_features
        
        self.resnet50.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet50(x)
        return x

class ResNet101(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(ResNet101, self).__init__()
        
        self.resnet101 = torchvision.models.resnet101(pretrained=isTrained)

        kernelCount = self.resnet101.fc.in_features
        
        self.resnet101.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)
        return x
      
class ResNet152(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(ResNet152, self).__init__()
        
        self.resnet152 = torchvision.models.resnet152(pretrained=isTrained)

        kernelCount = self.resnet152.fc.in_features
        
        self.resnet152.fc = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet152(x)
        return x

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, classCount)
        # self.mlp_2 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        # x = self.mlp_2(x)
        x = self.sigmoid(x)
        
        return x
 
class Inception_v3(nn.Module):

    def __init__(self, classCount, isTrained):
    
        super(Inception_v3, self).__init__()
        
        self.inception_v3 = nn.Sequential(*list(torchvision.models.inception_v3(pretrained=isTrained).children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount)  
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.inception_v3(x)
        x = self.avgpool(x)
        x = x.view(-1, 2048)
        x = self.bn(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x



class Inceptionresnetv2(nn.Module):

    def __init__(self, classCount):
    
        super(Inceptionresnetv2, self).__init__()
        
        self.model_name = 'inceptionresnetv2' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.inceptionresnetv2 = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(1536)      
        self.mlp = nn.Linear(1536, classCount)  
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.inceptionresnetv2(x)
        x = self.avgpool(x)
        x = x.view(-1, 1536)
        x = self.bn(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class InceptionV4(nn.Module):

    def __init__(self, classCount):
    
        super(InceptionV4, self).__init__()
        
        self.model_name = 'inceptionv4' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.inceptionv4 = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(1536)      
        self.mlp = nn.Linear(1536, classCount)  
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.inceptionv4(x)
        x = self.avgpool(x)
        x = x.view(-1, 1536)
        x = self.bn(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class DenseNet121_Inceptionv4(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_Inceptionv4, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()

        self.model_name = 'inceptionv4' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.inceptionv4 = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool_2 = nn.AdaptiveAvgPool2d(1)    

        self.mlp = nn.Linear(2560, 1000) 
        self.bn = nn.BatchNorm1d(2560) 
        self.mlp_1 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()    

    def forward(self, x):

        x_1 = self.densenet121(x)      
        x_1 = self.avgpool_1(x_1)

        x_2 = self.inceptionv4(x)
        x_2 = self.avgpool_2(x_2)

        cat_feature = torch.cat((x_1, x_2), 1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1)
        cat_feature = self.bn(cat_feature)
        cat_feature = self.mlp(cat_feature)
        cat_feature = self.mlp_1(cat_feature)
        output = self.sigmoid(cat_feature)
        
        return output

class ResNet18_ResNet34__Inceptionv4(nn.Module):
    
    def __init__(self, classCount, isTrained=False):
        
        super(ResNet18_ResNet34__Inceptionv4, self).__init__()        
        self.resnet18 = nn.Sequential(*list(torchvision.models.resnet18().children())[:-2])
        self.resnet34 = nn.Sequential(*list(torchvision.models.resnet34().children())[:-2])
        self.avgpool_1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_2 = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()

        self.model_name = 'inceptionv4' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.inceptionv4 = nn.Sequential(*list(self.model.children())[:-2])
        self.avgpool_3 = nn.AdaptiveAvgPool2d(1)    

        self.mlp = nn.Linear(2560, 1000) 
        self.bn = nn.BatchNorm1d(2560) 
        self.mlp_1 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()    

        
    def forward(self, x):
        
        x_1 = self.resnet18(x)      
        x_1 = self.avgpool_1(x_1)

        x_2 = self.resnet34(x)
        x_2 = self.avgpool_2(x_2)

        x_3 = self.inceptionv4(x)
        x_3 = self.avgpool_3(x_3)        

        # print(x_1.size(), x_2.size(), x_3.size())
        cat_feature = torch.cat((x_1, x_2, x_3), 1)

        cat_feature = cat_feature.view(cat_feature.size(0), -1)
        cat_feature = self.bn(cat_feature)
        cat_feature = self.mlp(cat_feature)
        cat_feature = self.mlp_1(cat_feature)
        output = self.sigmoid(cat_feature)
        
        return output

class polynet(nn.Module):

    def __init__(self, classCount):
    
        super(polynet, self).__init__()
        
        self.model_name = 'polynet' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.polynet = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.polynet(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class dpn92(nn.Module):

    def __init__(self, classCount):
        super(dpn92, self).__init__()
        
        self.model_name = 'dpn92' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet+5k')
        self.dpn92 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2688)      
        self.mlp = nn.Linear(2688, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn92(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

        
class dpn68b(nn.Module):

    def __init__(self, classCount):
        super(dpn68b, self).__init__()
        
        self.model_name = 'dpn68b' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet+5k')
        self.dpn68b = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(832)      
        self.mlp = nn.Linear(832, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn68b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class dpn131(nn.Module):

    def __init__(self, classCount):
        super(dpn131, self).__init__()
        
        self.model_name = 'dpn131' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.dpn131 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2688)      
        self.mlp = nn.Linear(2688, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn131(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class dpn107(nn.Module):

    def __init__(self, classCount):
        super(dpn107, self).__init__()
        
        self.model_name = 'dpn107' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet+5k')
        self.dpn107 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2688)      
        self.mlp = nn.Linear(2688, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn107(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class dpn68(nn.Module):

    def __init__(self, classCount):
        super(dpn68, self).__init__()
        
        self.model_name = 'dpn68' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.dpn68 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(832)      
        self.mlp = nn.Linear(832, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn68(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class dpn98(nn.Module):

    def __init__(self, classCount):
        super(dpn98, self).__init__()
        
        self.model_name = 'dpn98' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.dpn98 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2688)      
        self.mlp = nn.Linear(2688, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.dpn98(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class se_resnet152(nn.Module):

    def __init__(self, classCount):
        super(se_resnet152, self).__init__()
        
        self.model_name = 'se_resnet152' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.se_resnet152 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.se_resnet152(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class senet154(nn.Module):

    def __init__(self, classCount):
        super(senet154, self).__init__()
        
        self.model_name = 'senet154' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.senet154 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.senet154(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class se_resnet50(nn.Module):

    def __init__(self, classCount):
        super(se_resnet50, self).__init__()
        
        self.model_name = 'se_resnet50' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.se_resnet50 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.se_resnet50(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x        

class se_resnet101(nn.Module):

    def __init__(self, classCount):
        super(se_resnet101, self).__init__()
        
        self.model_name = 'se_resnet101' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.se_resnet101 = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.se_resnet101(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class se_resnext50_32x4d(nn.Module):

    def __init__(self, classCount):
        super(se_resnext50_32x4d, self).__init__()
        
        self.model_name = 'se_resnext50_32x4d' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.se_resnext50_32x4d = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.se_resnext50_32x4d(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class se_resnext101_32x4d(nn.Module):

    def __init__(self, classCount):
        super(se_resnext101_32x4d, self).__init__()
        
        self.model_name = 'se_resnext101_32x4d' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1000, pretrained='imagenet')
        self.se_resnext101_32x4d = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.bn = nn.BatchNorm1d(2048)      
        self.mlp = nn.Linear(2048, classCount) 
        self.sigmoid = nn.Sigmoid()   
    def forward(self, x):
        x = self.se_resnext101_32x4d(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class nasnetalarge(nn.Module):

    def __init__(self, classCount):
        super(nasnetalarge, self).__init__()
        
        self.model_name = 'nasnetalarge' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1001, pretrained='imagenet+background')
        self.nasnetalarge = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)     
        self.mlp = nn.Linear(4032, classCount) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.nasnetalarge(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

class pnasnet5large(nn.Module):

    def __init__(self, classCount):
        super(pnasnet5large, self).__init__()
        
        self.model_name = 'pnasnet5large' 
        self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1001, pretrained='imagenet+background')
        self.pnasnet5large = self.model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)     
        self.mlp = nn.Linear(4320, classCount) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.pnasnet5large(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x

# class nasnetalarge(nn.Module):

#     def __init__(self, classCount):
#         super(nasnetalarge, self).__init__()
        
#         self.model_name = 'nasnetalarge' 
#         self.model = pretrainedmodels.__dict__[self.model_name](num_classes=1001, pretrained='imagenet+background')
#         self.num_ftrs = self.model.last_linear.in_features
#         self.model.last_linear = nn.Linear(self.num_ftrs, classCount)
#         self.sigmoid = nn.Sigmoid()   

#     def forward(self, x):
#         x = self.model(x)
#         x = self.sigmoid(x)
#         return x

# ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']

def get_model(backbone):
    if backbone == 'DenseNet121':
        model = DenseNet121(2, True)
    elif backbone == 'DenseNet169':
        model = DenseNet169(2, True)  
    elif backbone == 'DenseNet161':
        model = DenseNet161(2, True) 
    elif backbone == 'DenseNet201':
        model = DenseNet201(2, True)       
    elif backbone == 'ResNet18':
        model = ResNet18(2, True)
    elif backbone == 'ResNet34':
        model = ResNet34(2, True)
    elif backbone == 'ResNet50':
        model = ResNet50(2, True)
    elif backbone == 'ResNet101':
        model = ResNet101(2, True)  
    elif backbone == 'ResNet152':
        model = ResNet152(2, True)
    elif backbone == 'Inception_v3':
        model = Inception_v3(2, True)   
    elif backbone == 'Inceptionresnetv2':
        model = Inceptionresnetv2(2)
    elif backbone == 'InceptionV4':
        model = InceptionV4(2) 
    elif backbone == 'dpn68':
        model = dpn68(2)
    elif backbone == 'dpn68b':
        model = dpn68b(2)
    elif backbone == 'dpn92':
        model = dpn92(2)
    elif backbone == 'dpn98':
        model = dpn98(2)
    elif backbone == 'dpn131':
        model = dpn131(2)
    elif backbone == 'dpn107':
        model = dpn107(2)
    elif backbone == 'polynet':
        model = polynet(2)
    elif backbone == 'senet154':
        model = senet154(2)
    elif backbone == 'se_resnet50':
        model = se_resnet50(2)
    elif backbone == 'se_resnet101':
        model = se_resnet101(2)
    elif backbone == 'se_resnet152':
        model = se_resnet152(2)
    elif backbone == 'se_resnext101_32x4d':
        model = se_resnext101_32x4d(2)
    elif backbone == 'se_resnext50_32x4d':
        model = se_resnext50_32x4d(2)
    elif backbone == 'nasnetalarge':
        model = nasnetalarge(2)
    elif backbone == 'pnasnet5large':
        model = pnasnet5large(2)
    elif backbone == 'DenseNet121_Inceptionv4':
        model = DenseNet121_Inceptionv4(2)
    elif backbone == 'ResNet18_ResNet34__Inceptionv4':
        model = ResNet18_ResNet34__Inceptionv4(2)

    return model

# 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 
# 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 
# 'se_resnet50', 'se_resnet101', 'se_resnet152', 
# 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131'
# 'InceptionV4', 