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
        project="cacd-ce2",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "ce",
            "dataset": "cacd",
            "epochs": num_epochs,
            }
    )
    
    NUM_CLASSES = 49
    BATCH_SIZE = 128
    GRAYSCALE = False
    


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)
    return imp


###################
# Dataset
###################

class CACDDataset(Dataset): #lectura del dataset (classe)
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None): 

        df = pd.read_csv(csv_path, index_col=0) #llegeix csv train, test o val
        self.img_dir = img_dir #directori de les imatges
        self.csv_path = csv_path #path del csv
        self.img_names = df['file'].values #nom de les imatges
        self.y = df['age'].values #edat
        self.transform = transform #transformacions

    def __getitem__(self, index): #rebre una imate al donar una posicio
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index])) #obrim la imatge

        if self.transform is not None:
            img = self.transform(img) #apliquem transformacions

        label = self.y[index] #guardem edat com label
        levels = [1]*label + [0]*(49 - 1 - label) #1 per fins la edat corresponent, 0 per les altres (hi han 49)

        levels = torch.tensor(levels, dtype=torch.float32) #a tensor
        
        return img, label, levels

    def __len__(self):
        return self.y.shape[0]

##########################
# MODEL
##########################


def conv_dw(in_channels, out_channels, stride=1):
    """Depthwise separable convolution"""
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MobileNetCORAL(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetCORAL, self).__init__()
        self.num_classes=num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv_dw(32, 64, stride=1),
            conv_dw(64, 128, stride=2),
            conv_dw(128, 128, stride=1),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256, stride=1),
            conv_dw(256, 512, stride=2),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 1024, stride=2),
            conv_dw(1024, 1024, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = MobileNetCORAL(num_classes)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)



def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    return mae, mse


if __name__ == '__main__':
    print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
    TRAIN_CSV_PATH = 'C:/Users/Usuario/Downloads/xnap-project-ed_group_13-main/xnap-project-ed_group_13-main/Starting point/datasets/cacd_train_sample.csv'
    VALID_CSV_PATH = 'C:/Users/Usuario/Downloads/xnap-project-ed_group_13-main/xnap-project-ed_group_13-main/Starting point/datasets/cacd_valid.csv'
    TEST_CSV_PATH = 'C:/Users/Usuario/Downloads/xnap-project-ed_group_13-main/xnap-project-ed_group_13-main/Starting point/datasets/cacd_test_sample.csv'
    IMAGE_PATH = 'C:/Users/Usuario/Downloads/DATASETS DDNN/CACD2000/'


    NUM_WORKERS = 4
    CUDA = 0
    SEED = 1
    IMP_WEIGHT = 0
    OUTPATH = 'afad-modelx'

    if CUDA >= 0:
        DEVICE = torch.device("cuda:%d" % CUDA)
    else:
        DEVICE = torch.device("cpu")

    if SEED == -1:
        RANDOM_SEED = None
    else:
        RANDOM_SEED = SEED

    IMP_WEIGHT = IMP_WEIGHT

    PATH = OUTPATH
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    LOGFILE = os.path.join(PATH, 'training.log')
    TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
    TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')

    # Logging

    header = []

    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Task Importance Weight: %s' % IMP_WEIGHT)
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
    if not IMP_WEIGHT:
        imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)
    elif IMP_WEIGHT == 1:
        imp = task_importance_weights(ages)
        imp = imp[0:NUM_CLASSES-1]
    else:
        raise ValueError('Incorrect importance weight parameter.')
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

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 





    start_time = time.time()

    best_mae, best_rmse, best_epoch = 999, 999, -1
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets, levels) in enumerate(train_loader):

            features = features.to(DEVICE)
            targets = targets
            targets = targets.to(DEVICE)
            levels = levels.to(DEVICE)

            # FORWARD AND BACK PROP
            logits, probas = model(features)
            cost = cost_fn(logits, levels, imp)
            optimizer.zero_grad()

            cost.backward()

            # UPDATE MODEL PARAMETERS
            optimizer.step()

            # LOGGING
            model.eval()

            if not batch_idx % 50:
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                    % (epoch+1, num_epochs, batch_idx,
                        len(train_dataset)//BATCH_SIZE, cost))
                print(s)
                with open(LOGFILE, 'a') as f:
                    f.write('%s\n' % s)

        # Wandb evaluation
        model.eval()
        with torch.set_grad_enabled(False):
            train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
            test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                    device=DEVICE)
            wandb.log({'epoch':epoch, 
                       'train_mae':train_mae, 'train_mse':train_mse,
                       'test_mae':test_mae, 'test_mse':test_mse})
            print(train_mse,test_mse)
        """
        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)
        #valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
        #                                          device=DEVICE)
        #if valid_mae < best_mae:
            #best_mae, best_rmse, best_epoch = valid_mae, torch.sqrt(valid_mse), epoch
            ########## SAVE MODEL #############
            #torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))
        
        
        #s = 'MAE/RMSE: | Current Valid: %.2f/%.2f Ep. %d | Best Valid : %.2f/%.2f Ep. %d' % (
        #    valid_mae, torch.sqrt(valid_mse), epoch, best_mae, best_rmse, best_epoch)
        #print(s)
        """
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
        #valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
        #                                        device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        #s = 'MAE/RMSE: | Train: %.2f/%.2f | Valid: %.2f/%.2f | Test: %.2f/%.2f' % (
            #train_mae, torch.sqrt(train_mse),
            #valid_mae, torch.sqrt(valid_mse),
            #test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)


    ########## EVALUATE BEST MODEL ######
    """
    model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
    model.eval()

    with torch.set_grad_enabled(False):
        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                                device=DEVICE)
        valid_mae, valid_mse = compute_mae_and_mse(model, valid_loader,
                                                device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)

        s = 'MAE/RMSE: | Best Train: %.2f/%.2f | Best Valid: %.2f/%.2f | Best Test: %.2f/%.2f' % (
            train_mae, torch.sqrt(train_mse),
            valid_mae, torch.sqrt(valid_mse),
            test_mae, torch.sqrt(test_mse))
        print(s)
        with open(LOGFILE, 'a') as f:
            f.write('%s\n' % s)

    """
    ########## SAVE PREDICTIONS ######
    all_pred = []
    all_probas = []
    with torch.set_grad_enabled(False):
        for batch_idx, (features, targets, levels) in enumerate(test_loader):
            
            features = features.to(DEVICE)
            logits, probas = model(features)
            all_probas.append(probas)
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
            lst = [str(int(i)) for i in predicted_labels]
            all_pred.extend(lst)

    torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
    with open(TEST_PREDICTIONS, 'w') as f:
        all_pred = ','.join(all_pred)
        f.write(all_pred)
    wandb.finish()

