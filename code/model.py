import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, roc_curve
import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from torch.hub import load
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-e', '--epochs', type = str)

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

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):

        print(f'{epoch = }:', end = '\t')

        for phase in ['train', 'test']:

            running_loss = 0

            dl_dict = {'train' : train_dataloader,
                        'test' : test_dataloader}

            dataloader = dl_dict[phase]

            if phase == 'train':

                model.train()

            else:

                model.eval()

            for i, data in enumerate(dataloader, 0):

                inputs, labels = data

                if phase == 'train':
                    
                    optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                running_loss += loss.item()

            # print(f'{running_loss = }')
            end = '\t' if phase == 'train' else '\n'
            
            print(f'{phase} loss = {running_loss / len(dataloader)}', end = end)

            if phase == 'train':

                train_losses.append(running_loss / len(dataloader))

            else:

                test_losses.append(running_loss / len(dataloader))

    print('Finished training')

    return train_losses, test_losses

def get_confusion_matrix(model, dataloader, threshold = 0.5):

    true = []
    pred = []

    model.eval()
    
    for data in tqdm(dataloader):

        inputs, labels = data

        true.append(labels.item())

        outputs = model(inputs)
        outputs = outputs.flatten().tolist()

        # curr_pred = 1 if outputs[1] > outputs[0] else 0
        curr_pred = 1 if outputs[1] > threshold else 0

        pred.append(curr_pred)

    conf_mat = confusion_matrix(true, pred)

    return conf_mat

def plot_pr_curve(model, dataloader):

    true = []
    pred = []

    model.eval()
    
    for data in tqdm(dataloader):

        inputs, labels = data

        true.append(labels.item())

        outputs = model(inputs)
        outputs = outputs.flatten().tolist()[1]

        pred.append(outputs)

    PrecisionRecallDisplay.from_predictions(true, pred)
    plt.show()

    precision, recall, thresholds = precision_recall_curve(true, pred)

    return precision, recall, thresholds

def get_roc_curve(model, dataloader):

    true = []
    pred = []

    model.eval()
    
    for data in tqdm(dataloader):

        inputs, labels = data

        true.append(labels.item())

        outputs = model(inputs)
        outputs = outputs.flatten().tolist()[1]

        pred.append(outputs)

    fpr, tpr, thresholds = precision_recall_curve(true, pred)

    return fpr, tpr, thresholds

        

#%% 

if __name__ == '__main__':

    args = parser.parse_args()

    print('Reading data...', end = '')
    df = pd.read_csv('./data/data_labeled.csv')
    print('Complete!')

    print((df.value_counts('simple') / df.shape[0]).to_dict())

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

    input_image = Image.open(f'./images/Bathroom/bath_17.jpg')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    output = model(input_batch)

    for param in model.parameters():

        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 2),
                            nn.Sigmoid())

    print(model(input_batch))

    weights_dict = (df.value_counts('simple') / df.shape[0]).to_dict()
    weights = torch.tensor([1 - weights_dict[0], 1 - weights_dict[1]])

    print(weights)

    criterion = nn.CrossEntropyLoss(weight = weights)
    optimizer = optim.SGD(model.parameters(), lr = 0.75)

    scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

    epochs = int(args.epochs)

    train_losses, test_losses = train_model(model, criterion, optimizer, scheduler, num_epochs = epochs)

    print(model(input_batch))

    train_pr, train_rc, train_thresholds = plot_pr_curve(model, train_dataloader)
    test_pr, test_rc, test_thresholds = plot_pr_curve(model, test_dataloader)

    # fpr, tpr, train_thresholds = get_roc_curve(model, train_dataloader)

    pr_rc_df = pd.DataFrame({'Precision' : train_pr[:-1],
                            'Recall' : train_rc[:-1],
                            'Thresholds' : train_thresholds})

    # roc_df = pd.DataFrame({'FPR' : fpr[:-1],
    #                         'TPR' : tpr[:-1],
    #                         'Thresholds' : train_thresholds})

    print(pr_rc_df)
    # print(roc_df)

    # thresholds = roc_df[roc_df.FPR < 0.25]
    # thresholds = min(thresholds['Thresholds'])

    thresholds = pr_rc_df[pr_rc_df.Precision >= 0.75][pr_rc_df.Recall >= 0.75]
    thresholds = min(thresholds['Thresholds'])

    print(thresholds)

    train_confusion = get_confusion_matrix(model, train_dataloader, threshold = thresholds)
    test_confusion = get_confusion_matrix(model, test_dataloader, threshold = thresholds)

    print(train_confusion)
    print(test_confusion)

    plt.plot(range(epochs), train_losses, label = 'Train')
    plt.plot(range(epochs), test_losses, label = 'Test')
    plt.show()

