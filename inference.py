#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy

from tqdm import tqdm
from PIL import Image
from PIL import ImageChops
from PIL import ImageEnhance
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler


# In[2]:


model = torch.load("model_ela.pt")


# In[3]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# In[4]:


model = model.to(device)


# In[5]:


class_names = ['fake', 'real']


# In[6]:


acc = torch.nn.Softmax(dim=1)


# In[7]:


# filename = '1.jpg'
def inference_img(filename):

    basename, extension = os.path.splitext(filename)
    resaved = 'resaved.jpg'
    ela = 'ela.png'
    im = Image.open(filename)
    im.save(resaved, 'JPEG', quality=90)
    resaved_im = Image.open(resaved)

    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    #     print('Maximum difference was {}'.format(max_diff))
    ela_im.save(ela)
    img_test = Image.open(ela)
    img_transforms = data_transforms(img_test)
    img_unsquueeze = img_transforms.unsqueeze(0).to(device)
    model.eval().to(device)
    output = model(img_unsquueeze)
    
    
    _, preds = torch.max(output, 1)
    return class_names[int(preds)], max(max(acc(output))).item()


# In[11]:


filename = 'D:/Code/Tima_Onbroading/ELA/datatest_private/fake/CMND MAT SAU 2.jpg'
label, score = inference_img(filename)
print(label)
print(score)


# In[ ]:




