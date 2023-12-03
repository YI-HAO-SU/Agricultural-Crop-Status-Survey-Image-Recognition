# -*- coding: utf-8 -*-
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
from tqdm import tqdm
from PIL import Image
import glob
import csv
import random
from torch.utils.data import random_split
import math
import wandb
import pandas as pd
from datetime import date
import torch.optim.lr_scheduler
from copy import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# +
local_dir = "/work/u2785625/AI_Cup_2022/ProcessedDatasets"
#Training details
batch_size = 88
shuffle = True
n_classes = 33
learning_rate = 0.01 #0.01
pretrained = True
n_epochs = 50
mergeVSetAndTSet = False

proportion_Train = 0.8
proportion_valid = 0.1

PATH_WEIGHTS = "/work/u2785625/AI_Cup_2022/Weights"

#Temprary Variables
numOfGPUs = 1


# -


experiment = wandb.init(project='AI_Cup_2022',name="AI_Cup_1125_ResNext50",resume='allow', anonymous='must')
experiment.config.update(dict(epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate ))

# +
train_tfm = transforms.Compose([ #train
    transforms.ToTensor(),
    transforms.Resize((456,456)),
    transforms.CenterCrop(456),
    transforms.Normalize([.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(90),        
    ])

test_tfm = transforms.Compose([ #valid and grading
    transforms.ToTensor(),
    transforms.Resize((456,456)),
    transforms.CenterCrop(456),
    transforms.Normalize([.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
    
])


# +
model = models.resnext50_32x4d(pretrained = pretrained)
#myModel = models.resnext50_32x4d(pretrained = pretrained)

#num_ftrs = model.classifier[1].in_features  #For efficientNet
#model.fc = nn.Linear(num_ftrs, n_classes)   #For efficientNet

num_ftrs = model.fc.in_features            #For resnext
model.fc = nn.Linear(num_ftrs, n_classes)    #For resnext


if torch.cuda.device_count() > 1:
    numOfGPUs = torch.cuda.device_count()
    model = nn.DataParallel(model)


# + endofcell="--"
#data = datasets.ImageFolder(root = local_dir)
#n = len(data)
#print(n)

#n_TrainData = math.floor(n * proportion_Train)

#n_ValidData = n - n_TrainData

#train_dataset, valid_dataset = random_split(data, [n_TrainData, n_ValidData])
#train_dataset.dataset = copy(data)

#train_dataset.dataset.transform = train_tfm
#valid_dataset.dataset.transform = test_tfm


#print('Number of training data : ',len(train_dataset))
#print('Number of validation data : ', len(valid_dataset))


##train_dataset, valid_dataset = random_split(
##    dataset=data,
##    lengths=[n_TrainData, n_ValidData],
##    generator=torch.Generator().manual_seed(0)
##)

#train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = numOfGPUs * 4)
#valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = numOfGPUs * 4)



# # +
data = datasets.ImageFolder(root = local_dir, transform = test_tfm)
n = len(data)
print(n)

n_TrainData = math.floor(n * proportion_Train)

n_ValidData = n - n_TrainData

print('Number of training data : ',n_TrainData)
print('Number of validation data : ', n_ValidData)


train_dataset, valid_dataset = random_split(
    dataset=data,
    lengths=[n_TrainData, n_ValidData],
    generator=torch.Generator().manual_seed(0)
)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = numOfGPUs * 4)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size, shuffle = shuffle, num_workers = numOfGPUs * 4)
# -
# --


train_loss_history = []
train_acc_history = []
valid_loss_history = []
valid_acc_history = []


def outPutFigure(epoch):
  #x = np.arange(epoch)
  #plt.figure(figsize=(15,10))
  #plt.plot(x, train_loss_history, label='train')
  #plt.plot(x, valid_loss_history, label='validation')
  #plt.legend()
  #plt.title('Training & Validation Loss')
  #plt.xlabel('Epochs')
  #plt.ylabel('Loss')
  #plt.savefig(f"/work/u2785625/AI_Cup_2022/ResultFigure/Loss_{epoch}.png")
  #print(f"Output Figure: {f"/work/u2785625/AI_Cup_2022/ResultFigure/Loss_{epoch}.png"}")
  x = np.arange(epoch)
  plt.figure(figsize=(15,10))
  plt.plot(x, _train_acc_history, label='train')
  plt.plot(x, validACCs, label='validation')
  plt.legend()
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.savefig("/work/u2785625/AI_Cup_2022/ResultFigures/Loss_70.png")
  print(f"Output Figure: /work/u2785625/AI_Cup_2022/ResultFigure/Acc_70.png")


# +
# Initialize a model, and put it on the device specified.
model = model.to(device)

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-5) 
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# The number of training epochs and patience.

patience = 300 # If no improvement in 'patience' epochs, early stop


# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0
global_step = 0

for epoch in range(n_epochs):
    since = time.time() # 記錄開始時間
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    for batch in tqdm(train_dataloader):
        global_step += 1
        
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()
        logits = model(imgs)


        optimizer.zero_grad()

        loss = criterion(logits, labels)


        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        optimizer.step()

        acc = (logits.argmax(dim=-1) == labels).float().mean()

        train_loss.append(loss.item())
        train_accs.append(acc)
        

    scheduler.step()  
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    print("train loss: " , train_loss)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc.cpu())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_dataloader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs, labels = imgs.cuda(), labels.cuda()
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs)

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels)

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    valid_loss_history.append(valid_loss)
    valid_acc_history.append(valid_acc)
    
    experiment.log({
        'epoch': epoch,
        'valid loss': valid_loss,
        'valid accuracy' : valid_acc,
        'training loss':train_loss,
        'training accuracy' : train_acc.cpu(),
        
        'learning rate': optimizer.param_groups[0]['lr'],
        }, step=global_step)



    if valid_acc > best_acc:
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} ---> best")
    else:
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:        
        best_acc = valid_acc
        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), PATH_WEIGHTS)
        else:
            torch.save(model.state_dict(), PATH_WEIGHTS)
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break
# -
_train_acc_history = list()
for item in train_acc_history:
    _train_acc_history.append(item.cpu())

outPutFigure(n_epochs)

# +
PATH_WEIGHTS = "/work/u2785625/AI_Cup_2022/Weights"
PUBLIC_TESTING_PATH = "/work/u2785625/AI_Cup_2022/PublicTesting/"
folderNames = ['0']#['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e']
OUTPUT_CSV_PATH = "/work/u2785625/AI_Cup_2022/PublicTesting_OutputCsv/"

local_dir = "/work/u2785625/AI_Cup_2022/Datasets"

model.eval()
List_FileName = list()
List_Output = list()
for folder in folderNames:
    folderPath = PUBLIC_TESTING_PATH + folder + '/'
    for fileName in glob.glob(folderPath + '*'):
        img = Image.open(fileName)
        x = test_tfm(img)
        x = x.unsqueeze(0)
        output = model(x)
        _, pred = torch.max(output, dim = 1)
        List_FileName.append(fileName)
        List_Output.append(data.classes[pred])
        print('FileName : ', fileName)
        print( 'Predict :　', data.classes[pred])
#img = Image.open(PATH_TO_IMAGE)  # Load image as PIL.Image
#x = transform(img)  # Preprocess image
#x = x.unsqueeze(0)  # Add batch dimension

#output = model(x)  # Forward pass
#pred = torch.argmax(output, 1)  # Get predicted class if multi-class classification
#print('Image predicted as ', pred)

# +
output_dict = {"file": List_FileName,
                "lebel": List_Output}
Out_dataframe = pd.DataFrame(output_dict)
print(Out_dataframe)

FileName = "test_beforeSave_"+ date.today().strftime("%m/%d").replace('/','') + '.csv'

Out_dataframe.to_csv(OUTPUT_CSV_PATH + FileName , encoding = 'utf-8', index = True)
