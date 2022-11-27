import pandas as pd
from sklearn.model_selection import train_test_split
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.hub import load
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#%% Functions

class ZappealingDataset(Dataset):

    def __init__(self, X):

        self.X = X

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        image = self.X[index]

        X = self.preprocess(image)

        return X

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):

    print('Beginning training')
    for epoch in range(num_epochs):

        print(f'{epoch = }:', end = '\t')

        # for phase in ['train', 'test']:

        running_loss = 0

        for i, data in enumerate(train_dataloader, 0):

            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # if i % 100 == 99:

            #     print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000}')

        # print(f'{running_loss = }')
        print(f'loss = {running_loss / len(train_dataloader)}')

        scheduler.step()

    print('Finished training')
        

#%% 

if __name__ == '__main__':

    print('Reading data...', end = '')
    df = pd.read_csv('./data/data_labeled.csv')
    print('Complete!')

    print('Loading model...', end = '')
    model = load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    print('Complete!')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Creating dataloader...', end = '')
    train_dataset = ImageFolder(root = 'data/images/train', transform = preprocess)
    test_dataset = ImageFolder(root = 'data/images/test', transform = preprocess)

    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
    print('Complete')

    print(df.head())

    # dataiter = iter(train_dataloader)
    # images, labels = next(dataiter)

    # print(labels)

    input_image = Image.open(f'./images/Bathroom/bath_1.jpg')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    output = model(input_batch)

    for param in model.parameters():

        param.requires_grad = False

    num_features = model.fc.in_features
    # model.fc = nn.Linear(num_features, 2)
    model.fc = nn.Sequential(nn.Linear(num_features, 2),
                            nn.Sigmoid())

    print(model(input_batch))

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)

    train_model(model, criterion, optimizer, scheduler)

