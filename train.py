#!/usr/bin/python3

#    Haris Ashraf
#    haris.ashraf@paccar.com
#    Version V1.0

# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
from PIL import Image
import torch

import json
import argparse




# Before you proceed, update the PATH
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"

# Import Category to flower name

# Command Line Arguments.

parser = argparse.ArgumentParser(
    description = 'Parser for predict function'
)
parser.add_argument('data_dir', action='store', default='./flowers/')
parser.add_argument('--save_dir', action="store", default='./checkpoint.pth')
parser.add_argument('--arch', action="store", default="vgg15")
parser.add_argument('--learning_rate', action="store", type=float,default=0.002)
parser.add_argument('--epochs', action="store", default=1, type=int)
parser.add_argument('--gpu', action="store", default="gpu")

parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=1024)

args = parser.parse_args()

data_dir = args.data_dir
checkpoint_path = args.save_dir
learning_rate = args.learning_rate

power = args.gpu
epochs = args.epochs
dropout = args.dropout
hidden_units = args.hidden_units

# GPU Available

# Use GPU if it's available
device = torch.device("cuda" if args.gpu=="gpu" else "cpu")
print("Cuda?", torch.cuda.is_available())

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def get_transforms():

# Define transforms for the training, validation, and testing sets

 
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

   valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

   test_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

   return train_transforms, valid_transforms, test_transforms 


# Load the datasets with ImageFolder

def load_data_set(train_transforms, valid_transforms, test_transforms):

   train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
   valid_data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
   test_data = datasets.ImageFolder(data_dir + '/test', transform=valid_transforms)

   # Using the image datasets and the trainforms, define the dataloaders

   trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
   validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
   testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

   return train_data, trainloader, validloader, testloader


def train_model(trainloader, validloader, testloader):

   if args.arch=='densenet121':
      model = models.densenet121(pretrained=True)
      input_length = 1024
   elif args.arch == 'alexnet':
      model = models.alexnet(pretrained=True)
      input_length = 9216
   else: 
      model = models.vgg16(pretrained=True)
      input_length = 25088

   print ('training architecture: ', args.arch)

   # Freeze parameters so we don't backprop through them
   for param in model.parameters():
       param.requires_grad = False

   from collections import OrderedDict
   classifier = nn.Sequential(OrderedDict([
                             ('fc1', nn.Linear(input_length, 1024)),
                             ('relu', nn.ReLU()),
                             ('fc2', nn.Linear(1024, 102)),
                             ('dropout', nn.Dropout(args.dropout)),
                             ('output', nn.LogSoftmax(dim=1))
                             ]))
    
   model.classifier = classifier

   criterion = nn.NLLLoss()

   # Only train the classifier parameters, feature parameters are frozen
   optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
   model.to(device)

   epochs = 2
   steps = 0
   running_loss = 0
   print_every = 10
   for epoch in range(epochs):
     for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(validloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

   return model, optimizer
   
# Save the checkpoint 
def save_checkpoint(model, train_data, optimizer, checkpoint_path):
   
   #Create Dictionary
   checkpoint = {
       'arch' : args.arch,
       'model_classifier': model.classifier,
       'model_state_dict': model.state_dict(),
       'train_data_class_to_idx': train_data.class_to_idx,   
       'optimizer_state_dict': optimizer.state_dict(),
       'dropout': args.dropout,
       'hidden_units': args.hidden_units
       
   }

   torch.save(checkpoint, checkpoint_path)
   print("checkpoint saved in file:", checkpoint_path)



def main():
   
   # Follow the steps outlined in the course. Name of the functions define the type of operation.

    train_transforms, valid_transforms, test_transforms = get_transforms() 

    train_data, trainloader, validloader, testloader = load_data_set (train_transforms, 
                                                                      valid_transforms, 
                                                                      test_transforms)

    model, optimizer = train_model (trainloader, validloader, testloader)

    save_checkpoint(model, train_data, optimizer, checkpoint_path)
 
if __name__ == "__main__":
    main()
