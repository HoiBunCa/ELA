#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[ ]:


# cd '/content/drive/MyDrive/Tima_Onbroading/ELA'


# In[ ]:


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# In[ ]:


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

# In[ ]:


model = torch.load("model_ela.pt")
model = model.to(device)
class_names = ['fake', 'real']
acc = torch.nn.Softmax(dim=1)


# In[ ]:


image_datasets = datasets.ImageFolder('/content/drive/MyDrive/Tima_Onbroading/ELA/datatest_private', data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True, num_workers=4)


# In[ ]:


y_true = []
y_pred = []
with torch.no_grad():
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        pred = model(inputs)
        print(labels.shape)
        print("---------------")
        _, preds = torch.max(pred, 1)

        for i1 in labels:
            y_true.append(int(i1))
        for i2 in preds:
            y_pred.append(int(i2))

# print(y_true)
# print(y_pred)


# In[ ]:


from sklearn.metrics import classification_report
target_names = ['real', 'fake']
print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_true, y_pred))

