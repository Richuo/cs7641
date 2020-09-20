# From https://github.com/brain-bzh/dcase-2020-task1-subtaskB

import os
import sys
import time
import copy

import random
import numpy as np

import torch
from torch.nn import functional as F

from utils import EarlyStopping

def train_model(model, hist, criterion, optimizer, dataloaders, dataset_sizes, 
                scheduler=None, num_epochs=25, estop=False, patience_es=5, device='cpu'):
    """
    Training function. 
    Return the trained model and a dictionary with the training info.
    """
    print("\n\n**TRAINING**\n")

    if estop: early_stopping = EarlyStopping(patience=patience_es, verbose=True, delta=0)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        lasttime = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            nb_batches = len(dataloaders[phase])

            # Iterate over data.
            #pbar = tqdm.tqdm([i for i in range(nb_batches)])
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                #pbar.update()
                #pbar.set_description("Processing batch %s" % str(batch_idx+1))  
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    #print(preds.size(), labels.size())

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #pbar.close()
            if phase == 'train' and scheduler != None and epoch != 0:
                scheduler.step(hist['val_loss'][-1])

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(running_corrects.double(), dataset_sizes[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))            

            hist[f'{phase}_loss'].append(epoch_loss)
            hist[f'{phase}_acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'val':
                valid_loss = epoch_loss # Register validation loss for Early Stopping

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print("Epoch complete in {:.1f}s\n".format(time.time() - lasttime))

        # Check Early Stopping
        if estop: early_stopping(valid_loss, model)
    
        if estop and early_stopping.early_stop:
            print("Early stopping")
            hist['best_val_acc'] = best_acc
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    best_acc = round(float(best_acc), 4)
    print('Best val Acc: {:4f}'.format(best_acc))
    hist['best_val_acc'] = best_acc
    hist['epochs'] = np.arange(epoch + 1)

    # load best model weights
    model.load_state_dict(best_model_wts)

    return (model, hist)


def test_model(model, hist, criterion, dataloaders, dataset_sizes, half=False, device='cpu'):
    """
    Testing function. 
    Print the loss and accuracy after the inference on the testset.
    """
    print("\n\n**TESTING**\n")

    sincetime = time.time()

    phase = "test"
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    list_y_pred = []
    list_y_true = []
    list_probs = []

    nb_batches = len(dataloaders[phase])

    #pbar = tqdm.tqdm([i for i in range(nb_batches)])

    # Iterate over data.
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
        #pbar.update()
        #pbar.set_description("Processing batch %s" % str(batch_idx+1))  
        inputs = inputs.to(device)
        labels = labels.to(device)

        # After Quantization
        if half:
            inputs = inputs.half()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = F.softmax(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        #print(preds)
        list_y_pred.append(int(preds.cpu()))
        list_y_true.append(int(labels.data.cpu()))
        list_probs.append(probs.cpu())

    #pbar.close()

    test_loss = running_loss / dataset_sizes[phase]
    test_acc = running_corrects.double() / dataset_sizes[phase]
    test_acc = round(float(test_acc), 4)
    hist['test_acc'] = test_acc

    hist['y_pred'] = list_y_pred
    hist['probs'] = list_probs
    hist['y_true'] = list_y_true

    print('\nTest stats -  Loss: {:.4f} Acc: {:.2f}%'.format(test_loss, test_acc*100))            

    print("Inference on Testset complete in {:.1f}s\n".format(time.time() - sincetime))

    return hist
