from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import torch.utils.data as data
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations

import json
fpath = open('../configs/path_configs_25.json', encoding='utf-8')
path_data = json.load(fpath)
train_img_path = path_data['train_img_path']

IMAGENET_SIZE = 256

train_transform = albumentations.Compose([
        
    albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
    # albumentations.RandomCrop(512, 512),
        # illumilation
    # albumentations.JpegCompression(quality_lower=99, quality_upper=100,p=0.5),
    albumentations.OneOf([
        albumentations.RandomGamma(gamma_limit=(60, 120), p=0.9),
        albumentations.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.9),
        albumentations.RandomBrightness(limit=0.2, p=0.9),
        albumentations.RandomContrast(limit=0.2, p=0.9)
        ]),
#                                    CLAHE(clip_limit=4.0, tile_grid_size=(3, 3), p=1)
    # albumentations.GaussNoise(var_limit=(10, 30), p=0.5),
    
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT, p=1),
#                                        OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=1)
    albumentations.OneOf([
        albumentations.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
        albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1)
    ], p=0.5),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

val_transform = albumentations.Compose([
    albumentations.Resize(IMAGENET_SIZE, IMAGENET_SIZE),
    # albumentations.CenterCrop(512, 512),
    # albumentations.JpegCompression(quality_lower=99, quality_upper=100,p=1),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

def change_name(name):
    if 'local_data' in name:
        new_name = name.split('/')[-1]
    else:
        name_split_list = name.split('/')
        new_name = name_split_list[1] + '_'+ name_split_list[2] + '_'+ name_split_list[3] + '_'+ name_split_list[4]
    return new_name

class Chexnet_dataset_10(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        # print(name)
        label = torch.FloatTensor(self.df[self.df['Path']==(name) ].loc[:, 'var_0':'var_9'].values)
        # label = torch.FloatTensor(self.df[self.df['Image Index']==(name) ].loc[:, 'var_0':'var_24'].values)
        name = change_name(name)
        # print(name)
        image = cv2.imread(train_img_path + name)
        

        image = self.transform(image=image)['image'].transpose(2, 0, 1)
        # print(image.shape)

        return image, label

class Chexnet_dataset_25(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        # print(name)
        # label = torch.FloatTensor(self.df[self.df['Path']==(name) ].loc[:, 'var_0':'var_9'].values)
        label = torch.FloatTensor(self.df[self.df['Image Index']==(name) ].loc[:, 'var_0':'var_24'].values)
        # name = change_name(name)
        # print(name)
        image = cv2.imread(train_img_path + name)
        

        image = self.transform(image=image)['image'].transpose(2, 0, 1)
        # print(image.shape)

        return image, label

class Chexnet_dataset_chexpert(data.Dataset):
    def __init__(self,
                 df = None,
                 name_list = None,
                 transform = None
                 ):
        self.df = df
        self.name_list = name_list
        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        # print(name)
        # label = torch.FloatTensor(self.df[self.df['Path']==(name) ].loc[:, 'var_0':'var_9'].values)
        label = self.df[self.df['Path']==(name) ].loc[:, 'Enlarged Cardiomediastinum':'Support Devices'].values
        name = change_name(name)
        # print(name)
        image = cv2.imread(train_img_path + name)
        

        image = self.transform(image=image)['image'].transpose(2, 0, 1)
        label = np.nan_to_num(label)
        label[label==-1] = 1
        label = torch.FloatTensor(label)
        # print(label)
        # print(image.shape)

        return image, label

def generate_dataset_loader_10(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Chexnet_dataset_10(df_all, c_train, train_transform)
    val_dataset = Chexnet_dataset_10(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
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

    return train_loader, val_loader

def generate_dataset_loader_25(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Chexnet_dataset_25(df_all, c_train, train_transform)
    val_dataset = Chexnet_dataset_25(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
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

    return train_loader, val_loader

def generate_dataset_loader_chexpert(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers):

    train_dataset = Chexnet_dataset_chexpert(df_all, c_train, train_transform)
    val_dataset = Chexnet_dataset_chexpert(df_all, c_val, val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,        
        shuffle=True,
        # sampler=ImbalancedDatasetSampler(train_dataset),
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

    return train_loader, val_loader


