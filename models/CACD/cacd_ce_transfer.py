# coding: utf-8

#############################################
# Consistent Cumulative Logits with ResNet-34
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

import wandb

torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    ##########################
    # SETTINGS
    ##########################

    # Hyperparameters
    learning_rate = 0.0005
    num_epochs = 2
    wandb.init(
    # set the wandb project where this run will be logged
        project="cacd-ce",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "tranfer-ce",
            "dataset": "cacd",
            "epochs": num_epochs,
            }
    )
    
    NUM_CLASSES = 49
    BATCH_SIZE = 256
    GRAYSCALE = False
    




###################
# Dataset
###################

class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]



##########################
# MODEL
##########################



def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = resnet50(weights = "IMAGENET1K_V2")
    model.fc = nn.Linear(2048, num_classes)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    return torch.mean(val)


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        probas = torch.softmax(logits, dim=1)
        #logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


if __name__ == '__main__':
    TRAIN_CSV_PATH = 'datasets/cacd_train.csv'
    VALID_CSV_PATH = 'datasets/cacd_valid.csv'
    TEST_CSV_PATH = 'datasets/cacd_test.csv'
    IMAGE_PATH = '../../../../Desktop/Datasets/CACD2000/'


    NUM_WORKERS = 4
    CUDA = -1
    SEED = 1
    OUTPATH = 'cacd-ordinal'

    if CUDA >= 0:
        DEVICE = torch.device("cuda:%d" % CUDA)
    else:
        DEVICE = torch.device("cpu")

    if SEED == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = SEED


    PATH = OUTPATH
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    LOGFILE = os.path.join(PATH, 'training.log')
    TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
    TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')
    COSTES = os.path.join(PATH, 'costs.log')

    # Logging

    header = []

    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Output Path: %s' % PATH)
    header.append('Script: %s' % sys.argv[0])

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()




    # Architecture
    

    df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
    ages = df['age'].values
    del df
    ages = torch.tensor(ages, dtype=torch.float)





    # Data-specific scheme

    imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)

    imp = imp.to(DEVICE)




    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.RandomCrop((120, 120)),
                                           transforms.ToTensor()])
    
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform)
    
    
    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.CenterCrop((120, 120)),
                                           transforms.ToTensor()])
    
    test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               transform=custom_transform2)
    
    valid_dataset = CACDDataset(csv_path=VALID_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform2)
    
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS)
    
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=NUM_WORKERS)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)



    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    model = resnet34(NUM_CLASSES, GRAYSCALE)
    set_parameter_requires_grad(model, True)

    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True
    model.to(DEVICE)
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

    optimizer = torch.optim.Adam(params_to_update, lr=learning_rate) 


    def compute_mae_and_mse(model, data_loader, device):
        mae, mse, num_examples = 0., 0., 0
        for i, (features, targets) in enumerate(data_loader):
                
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            probas = torch.softmax(logits, dim=1)
            #logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            mae += torch.sum(torch.abs(predicted_labels - targets))
            mse += torch.sum((predicted_labels - targets)**2)
        
        mae = mae.float()/num_examples
        mse = mse.float()/num_examples
        
        return mae, mse


    start_time = time.time()

    best_mae, best_rmse, best_epoch = 999, 999, -1
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets.to(DEVICE)

            # FORWARD AND BACK PROP

            logits = model(features)
            probas = torch.softmax(logits, dim=1)
            #logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            if not batch_idx % 50:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        len(train_dataset)//BATCH_SIZE, cost))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)

        model.eval()
        with torch.set_grad_enabled(False):
            train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
            test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                    device=DEVICE)
            wandb.log({'epoch':epoch, 
                       'train_mae':train_mae, 'train_mse':train_mse,
                       'test_mae':test_mae, 'test_mse':test_mse})
        
        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                  device=DEVICE)

        if test_mae < best_mae:
            best_mae, best_rmse, best_epoch = test_mae, torch.sqrt(test_mse), epoch
            ########## SAVE MODEL #############
            torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


        s = 'MAE/RMSE: | Current Test: %.2f/%.2f Ep. %d | Best Test : %.2f/%.2f Ep. %d' % (
            test_mae, torch.sqrt(test_mse), epoch, best_mae, best_rmse, best_epoch)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

        s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):  # save memory during inference

        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (
            train_mae, torch.sqrt(train_mse), test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Best MAE: %.2f | Best RMSE: %.2f | Best Epoch: %d' % (best_mae, best_rmse, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    ########## SAVE PREDICTIONS ######

    model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
    model.eval()
    all_pred = []
    with torch.set_grad_enabled(False):
        for batch_idx, (features, targets) in enumerate(test_loader):
            
            features = features.to(DEVICE)
            logits = model(features)
            probas = torch.softmax(logits, dim=1)
            #logits, probas = model(features)
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
            lst = [str(int(i)) for i in predicted_labels]
            all_pred.extend(lst)

    with open(TEST_PREDICTIONS, 'w') as f:
        all_pred = ','.join(all_pred)
        f.write(all_pred)