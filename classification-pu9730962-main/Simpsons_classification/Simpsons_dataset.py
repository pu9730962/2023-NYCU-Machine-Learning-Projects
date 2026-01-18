import pandas as pd
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import os
from torchvision import transforms

def getdata(mode):
    if mode == "train":
        train_data_name=pd.read_csv("train_img.csv",header=None)
        train_label_name=pd.read_csv("train_label.csv",header=None)
        return np.squeeze(train_data_name.values),np.squeeze(train_label_name.values)
    elif mode =="valid":
        valid_data_name=pd.read_csv("valid_img.csv",header=None)
        valid_label_name=pd.read_csv("valid_label.csv",header=None)
        return np.squeeze(valid_data_name.values),np.squeeze(valid_label_name.values)
    else:
        test_data_name=pd.read_csv("test_img.csv",header=None)
        return np.squeeze(test_data_name.values)

class train(Dataset):
    def __init__(self):
        self.data_root="C:/test/Simpsons_CNN/train_data/"
        self.train_data_name,self.train_label_name=getdata("train")
        self.transformation=transforms.Compose([transforms.RandomChoice([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomApply([transforms.RandomChoice([transforms.RandomResizedCrop(size=224, scale=(0.08, 1)),transforms.ColorJitter(brightness=0.3), transforms.ColorJitter(contrast=0.3),transforms.ColorJitter(saturation=0.3)])],p=0.5),transforms.RandomPerspective(distortion_scale=0.2,p=0.5),transforms.RandomRotation(degrees=20),transforms.RandomGrayscale(p=0.5)]),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #self.transformation=transforms.Compose([transforms.RandomChoice( [ 
                                # transforms.RandomHorizontalFlip(p=0.5),
                                # transforms.ColorJitter(contrast=0.9),
                                # transforms.ColorJitter(brightness=0.1),
                                # transforms.RandomApply( [ transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(contrast=0.9) ], p=0.5),
                                # transforms.RandomApply( [ transforms.RandomHorizontalFlip(p=1), transforms.ColorJitter(brightness=0.1) ], p=0.5),
                                #             ] ),
                                # transforms.ToTensor(),
                                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.train_data_name)

    def __getitem__(self,index):
        train_data=self.transformation(Image.open(self.data_root+f'{self.train_data_name[index]}'))
        train_label=self.train_label_name[index]
        return train_data,train_label

class valid(Dataset):
    def __init__(self):
        self.data_root="C:/test/Simpsons_CNN/train_data/"
        self.valid_data_name,self.valid_label_name=getdata("valid")
        self.transformation=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.valid_data_name)

    def __getitem__(self,index):
        valid_data=self.transformation(Image.open(self.data_root+f'{self.valid_data_name[index]}'))
        valid_label=self.valid_label_name[index]
        return valid_data,valid_label

class test(Dataset):
    def __init__(self):
        self.data_root="C:/test/Simpsons_CNN/test_data/"
        self.test_data_name=getdata("test")
        self.transformation=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def __len__(self):
        return len(self.test_data_name)

    def __getitem__(self,index):
        test_data=self.transformation(Image.open(self.data_root+f'{self.test_data_name[index]}'))
        return test_data
        

    


