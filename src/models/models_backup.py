
class DenseNet121_change_avg_efficient(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_change_avg_efficient, self).__init__()
        
        model = DenseNet(
            growth_rate=12,
            block_config=[(121 - 4) // 6 for _ in range(3)],
            num_classes=10,
            small_inputs=True,
            efficient=True,
            )
        self.densenet121 = model.features
        # self.densenet121[0] = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
        # #                                     # nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # #                                     # nn.ReLU(inplace=True),
        #                                     self.densenet121[0])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(405, classCount)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 405)
        x = self.mlp(x)
        x = self.sigmoid(x)
        
        return x



class DenseNet121_change_avg_new(nn.Module):

    def __init__(self, classCount, isTrained=False):
    
        super(DenseNet121_change_avg_new, self).__init__()
        
        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        # self.densenet121[0] = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
        # #                                     # nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        # #                                     # nn.ReLU(inplace=True),
        #                                     self.densenet121[0])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, classCount)
        self.sigmoid = nn.Sigmoid()   
        self.matrix = nn.Linear(classCount, classCount)

    def forward(self, x):

        x = self.densenet121(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        x_relate = self.matrix(x)
        x_relate = self.sigmoid(x_relate)
        
        return x_relate

class DenseNet121_change_avg_MCNN(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121_change_avg_MCNN, self).__init__()
        
        kernelCount = 512

        self.densenet121_512 = torchvision.models.densenet121(pretrained=isTrained).features
        self.densenet121_512_avgpool = nn.AdaptiveAvgPool2d(1)  

        self.densenet121_256 = torchvision.models.densenet121(pretrained=isTrained).features
        self.densenet121_256_avgpool = nn.AdaptiveAvgPool2d(1)  

        self.densenet121_128 = torchvision.models.densenet121(pretrained=isTrained).features
        self.densenet121_128_avgpool = nn.AdaptiveAvgPool2d(1)  

        # self.densenet121_64 = torchvision.models.densenet121(pretrained=isTrained).features
        # self.densenet121_64_avgpool = nn.AdaptiveAvgPool2d(1)  

        self.bn_1 = nn.BatchNorm1d(1024)
        self.bn_2 = nn.BatchNorm1d(1024)
        self.bn_3 = nn.BatchNorm1d(1024)
        self.bn_4 = nn.BatchNorm1d(1024*3)
        self.bn_5 = nn.BatchNorm1d(1000)

        self.mlp_1 = nn.Linear(1024*3, 1000)

        self.mlp_2 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.max_pool_4 = nn.MaxPool2d(4, 4)
        # self.max_pool_8 = nn.MaxPool2d(8, 8)


    def forward(self, x):

        x_1 = self.densenet121_512(x)
        x_1 = self.relu(x_1)
        x_1 = self.densenet121_512_avgpool(x_1)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1 = self.bn_1(x_1)

        x_2 = self.max_pool_2(x)
        x_2 = self.densenet121_256(x_2)
        x_2 = self.relu(x_2)
        x_2 = self.densenet121_256_avgpool(x_2)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2 = self.bn_2(x_2)

        x_3 = self.max_pool_4(x)
        x_3 = self.densenet121_128(x_3)
        x_3 = self.relu(x_3)
        x_3 = self.densenet121_128_avgpool(x_3)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = self.bn_3(x_3)

        # x_4 = self.max_pool_8(x)
        # x_4 = self.densenet121_64(x)
        # x_4 = self.densenet121_64_avgpool(x_1)
        # x_4 = x_4.view(x_4.size(0), -1)

        # print(x_1.shape, x_2.shape, x_3.shape, x_4.shape)
        out = torch.cat((x_1, x_2, x_3), 1)
        out = self.bn_4(out)
        out = self.mlp_1(out)
        out = self.bn_5(out)
        out = self.mlp_2(out)
        out = self.sigmoid(out)

        return out


class DenseNet121_change_avg_GAP(nn.Module):

    def __init__(self, classCount, isTrained):
        super(DenseNet121_change_avg_GAP, self).__init__()

        self.densenet121 = torchvision.models.densenet121(pretrained=isTrained).features
        self.densenet121_avgpool = nn.AdaptiveAvgPool2d(1)  

        # 0:4  4:6  6:8  8:10  10:11
        self.densenet121_feature_1 = self.densenet121[0:4]
        self.densenet121_feature_2 = self.densenet121[4:6]
        self.densenet121_feature_3 = self.densenet121[6:8]
        self.densenet121_feature_4 = self.densenet121[8:10]
        self.densenet121_feature_5 = self.densenet121[10:11]

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(1024)

        self.bn_6 = nn.BatchNorm1d(1984)
        self.bn_7 = nn.BatchNorm1d(1000)
        self.bn_8 = nn.BatchNorm1d(1000)
        self.mlp_1 = nn.Linear(1984, 1000)
        self.mlp_2 = nn.Linear(1000, 1000)
        self.mlp_3 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.densenet121_feature_1(x)
        x_1 = self.densenet121_avgpool(x)
        x_1 = x_1.view(x_1.size(0), -1)
        x_1 = self.bn_1(x_1)

        x = self.densenet121_feature_2(x)
        x_2 = self.densenet121_avgpool(x)
        x_2 = x_2.view(x_2.size(0), -1)
        x_2 = self.bn_2(x_2)

        x = self.densenet121_feature_3(x)
        x_3 = self.densenet121_avgpool(x)
        x_3 = x_3.view(x_3.size(0), -1)
        x_3 = self.bn_3(x_3)

        x = self.densenet121_feature_4(x)
        x_4 = self.densenet121_avgpool(x)
        x_4 = x_4.view(x_4.size(0), -1)
        x_4 = self.bn_4(x_4)

        x = self.densenet121_feature_5(x)
        x_5 = self.densenet121_avgpool(x)
        x_5 = x_5.view(x_5.size(0), -1)
        x_5 = self.bn_5(x_5)


        out = torch.cat((x_1, x_2, x_3, x_4, x_5), 1)
        out = self.bn_6(out)
        out = self.mlp_1(out)
        out = self.bn_7(out)
        out = self.mlp_2(out)
        out = self.bn_8(out)
        out = self.mlp_3(out)
        out = self.sigmoid(out)

        return out

class ResNet101(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101()
        self.resnet101.conv1 = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
                                             nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            self.resnet101.conv1)
        self.resnet101.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet101.fc = nn.Sequential(nn.Linear(2048, classCount),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnet101(x)      
        return x



class ResNet34(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet34, self).__init__()
        
        self.resnet34 = torchvision.models.resnet34(pretrained=isTrained)
        self.resnet34.conv1 = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
                                             nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            self.resnet34.conv1)
        self.resnet34.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet34.fc = nn.Sequential(nn.Linear(512, classCount),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.resnet34(x)
        return x

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet18, self).__init__()
        
        self.resnet18 = torchvision.models.resnet18(pretrained=isTrained)
        self.resnet18.conv1 = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
                                            self.resnet18.conv1)
        # self.resnet18.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        kernelCount = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(self.resnet18.fc,
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(1000, 1000),
                                        nn.Dropout(0.5),
                                        nn.ReLU(),
                                        nn.Linear(1000, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.resnet18(x)
        return x

class ResNet18_MCNN(nn.Module):

    def __init__(self, classCount, isTrained):
        super(ResNet18_MCNN, self).__init__()
        
        kernelCount = 512

        self.resnet18_512 = torchvision.models.resnet18(pretrained=isTrained)
        self.resnet18_512.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet18_512 = nn.Sequential(*list(self.resnet18_512.children())[:-1])

        self.resnet18_256 = torchvision.models.resnet18(pretrained=isTrained)
        self.resnet18_256.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.resnet18_256 = nn.Sequential(*list(self.resnet18_256.children())[:-1]) 

        self.resnet18_128 = torchvision.models.resnet18(pretrained=isTrained)
        self.resnet18_128.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet18_128 = nn.Sequential(*list(self.resnet18_128.children())[:-1])
        
        self.resnet18_64 = torchvision.models.resnet18(pretrained=isTrained)
        self.resnet18_64.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.resnet18_64 = nn.Sequential(*list(self.resnet18_64.children())[:-1])

        self.bn_1 = nn.BatchNorm2d(2048)
        self.mlp_1 = nn.Linear(2048, 1000)

        self.mlp_2 = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()
        
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.max_pool_4 = nn.MaxPool2d(4, 4)
        self.max_pool_8 = nn.MaxPool2d(8, 8)


    def forward(self, x):

        x_1 = self.resnet18_512(x)
        x_1 = x_1.view(x_1.size(0), -1)

        x_2 = self.max_pool_2(x)
        x_2 = self.resnet18_256(x_2)
        x_2 = x_2.view(x_2.size(0), -1)

        x_3 = self.max_pool_4(x)
        x_3 = self.resnet18_128(x_3)
        x_3 = x_3.view(x_3.size(0), -1)

        x_4 = self.max_pool_8(x)
        x_4 = self.resnet18_64(x_4)
        x_4 = x_4.view(x_4.size(0), -1)

        # print(x_1.shape, x_2.shape, x_3.shape, x_4.shape)
        out = torch.cat((x_1, x_2, x_3, x_4), 1)
        # out = self.bn_1(out)
        out = self.mlp_1(out)
        out = self.mlp_2(out)
        out = self.sigmoid(out)

        return out

class Pnasnet5large(nn.Module):

    def __init__(self, classCount):
        super(Pnasnet5large, self).__init__()
        
        self.pnasnet5large = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1001, pretrained='imagenet+background')
        self.pnasnet5large.conv_0.conv = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
                                            self.pnasnet5large.conv_0.conv)
        num_ftrs = self.pnasnet5large.last_linear.in_features
        self.pnasnet5large.last_linear = nn.Sequential(nn.Linear(num_ftrs, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.pnasnet5large(x)
        return x

class InceptionV3(nn.Module):

    def __init__(self, classCount, isTrained):
        super(InceptionV3, self).__init__()
        self.inceptionv3 = nn.Sequential(*list(torchvision.models.inception_v3().children())[:-2])
        self.inceptionv3[0] = nn.Sequential(nn.Conv2d(4, 3, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False), 
                                             nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                            self.inceptionv3[0])
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(nn.Linear(2048, classCount),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.inceptionv3(x)  
        
        x = self.avgpool(x)
        x = F.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class Inceptionv3_GAP(nn.Module):
    def __init__(self, classCount, isTrained):
        super().__init__()

        model_ft = torchvision.models.inception_v3(pretrained=isTrained)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.AuxLogits = torchvision.models.inception.InceptionAux(768, classCount)
        self.fc1 = nn.Linear(2048+64+192, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.classifier = nn.Linear(1000, classCount)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        ginp1 = F.dropout(x, p=0.25)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        ginp2 = F.dropout(x, p=0.25)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        if self.training:
            aux = self.AuxLogits(x)
            aux = self.sigmoid(aux)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)

        ginp3 = F.dropout(x, p=0.25)
        ginp3 = self.avgpool(ginp3)
        ginp3 = ginp3.view(-1, 2048)

        ginp1 = self.avgpool(ginp1)
        ginp1 = ginp1.view(ginp3.shape[0], -1)

        ginp2 = self.avgpool(ginp2)
        ginp2 = ginp2.view(ginp3.shape[0], -1)

        x = torch.cat((ginp1, ginp2, ginp3), 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        x = self.classifier(x)
        x = self.sigmoid(x)
        if self.training:
            return x, aux
        return x
