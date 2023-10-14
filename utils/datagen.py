import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np

img_size = 256

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, df, is_train, sup, transform=None):
        self.dataset_path = dataset_path
        self.df = df
        self.is_train = is_train
        self.sup = sup
        if self.sup == True:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
            ])
        else: # if unsupervised, apply random transformation too
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_path = row['source']
        target_path = row['target']
        source_img = cv2.imread(source_path, 0)
        target_img = cv2.imread(target_path, 0)
        source_img = cv2.resize(source_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        target_img = cv2.resize(target_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # convert images to float32
        source_img = (source_img/np.max(source_img)).astype(np.float32)
        target_img = (target_img/np.max(target_img)).astype(np.float32)
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)
        if self.sup:
            affine_params = np.array([row['M[0]'], row['M[1][0]'], row['M[1][1]'], row['M[2]'], row['M[3][0]'], row['M[3][1]']]).astype(np.float32)
            return source_img, target_img, affine_params 
        else:
            return source_img, target_img
        
        

def datagen(dataset, is_train, sup):
    if dataset == 0:
        # actual eye dataset
        dataset_path = 'Dataset/Dataset-processed'
        if is_train:
            df = pd.read_csv('Dataset/dataset_eye_actual.csv')
            
            # count number of rows that df['training'] == 1
            print('Training eye dataset')
            print('Number of training data: ', len(df[df['training'] == 1]))

            df = df[df['training'] == 1]
        else:
            df = pd.read_csv('Dataset/dataset_eye_actual.csv')

            # count number of rows that df['training'] == 0
            print('Test eye dataset')
            print('Number of testing data: ', len(df[df['training'] == 0]))

            df = df[df['training'] == 0]

    elif dataset == 1:
        if is_train:
            # synthetic eye dataset
            dataset_path = 'Dataset/synthetic_eye_dataset_train'
            df = pd.read_csv('Dataset/dataset_eye_synth_train.csv')
        else:
            # synthetic eye dataset
            dataset_path = 'Dataset/synthetic_eye_dataset_test'
            df = pd.read_csv('Dataset/dataset_eye_synth_test.csv')
            
    elif dataset == 2:
        # synthetic shape dataset
        dataset_path = 'Dataset/synthetic_shape_dataset'
        if is_train:
            df = pd.read_csv('Dataset/dataset_shape_synth_train.csv')
        else:  
            df = pd.read_csv('Dataset/dataset_shape_synth_test.csv')
    else:
        raise ValueError('Invalid dataset parameter')

    dataset = MyDataset(dataset_path, df, is_train, sup)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=is_train)

    return dataloader

