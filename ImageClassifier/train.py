import train_args
import sys
import os
import json
import torch
from torch import nn, optim
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

def main():
    parser = train_args.get_args()
    cli = parser.parse_args()
    with open('categories_json', 'r') as f:
        cat_to_name = json.load(f)

    output_size = len(cat_to_name)
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    training_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),transforms.RandomRotation(25),transforms.RandomGrayscale(p=0.02),transforms.RandomResizedCrop(224),transforms.ToTensor(),transforms.Normalize(means,std)])
    data_dir= 'flowers'
    train_dir = data_dir + '/train'
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    model = models.vgg16(pretrained=True)
    input_size = 0
    input_size = model.classifier[0].in_features
    for param in model.parameters():
        param.requires_grad = False
    ord_dict = OrderedDict()
    hidden_units=[3136,784]
    hidden_units.insert(0, input_size)

    print("hidden layer:",hidden_units)

    for j in range(len(hidden_units) - 1):
        ord_dict['fc' + str(j + 1)] = nn.Linear(hidden_units[j], hidden_units[j + 1])
        ord_dict['relu' + str(j + 1)] = nn.ReLU()
        ord_dict['dropout' + str(j + 1)] = nn.Dropout(p=0.15)

    ord_dict['output'] = nn.Linear(hidden_units[j + 1], output_size)
    ord_dict['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(ord_dict)
    model.classifier = classifier
    model.zero_grad()
    criterion = nn.NLLLoss()
    lr=0.001
    print("learning rate" ,lr)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    device = torch.device("cpu")

    if cli.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Using CPU.")

    print("Device:",device)
    model = model.to(device)

    batch_size=train_dataloader.batch_size
    check = 100
    print(f'Training {len(train_dataloader.batch_sampler)} batch of {batch_size}.')
    for i in range(cli.epochs):
        print("Epoch:",i+1)
        loss = 0
        total = 0
        correct = 0
        prev_check = 0
       
        for k, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            itr = (k + 1)
            if itr % check == 0:
                avg_loss = loss/itr
                accuracy =(correct/total) * 100
                print("loss:",avg_loss)
                print("accuracy",accuracy)
                prev_check = (k + 1)
    model.class_to_idx = train_data.class_to_idx
    model_state = {'epoch': cli.epochs,'class_to_idx': model.class_to_idx,'state_dict': model.state_dict(),'optimizer_dict': optimizer.state_dict(),'classifier': model.classifier,'arch': 'vgg16'}
    save_location = f'{cli.save_dir}/{cli.save_name}.pth'
    print("Checkpoint saved at:",save_location)
    torch.save(model_state, save_location)

