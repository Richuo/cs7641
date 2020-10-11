import os
import sys
import time
import copy

import random
import numpy as np

from sklearn.metrics import log_loss

import torch
from torch.nn import functional as F

from utils import EarlyStopping

def train_model(model, hist, criterion, optimizer, dataloaders, dataset_sizes, max_iterations=2000, 
                scheduler=None, estop=False, patience_es=5, device='cpu', do_print=False):
    """
    Training function. 
    Return the trained model and a dictionary with the training info.
    """
    if do_print: print("\n\n**TRAINING**\n")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    lasttime = time.time()
    if do_print: print('-' * 10)

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
            if max_iterations < 0:
                break
            max_iterations -= len(inputs)


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

                current_time = time.time()-since

                hist[f'train_ite'].append(test_model_ite(model=model, criterion=criterion, 
                                                            running_time=current_time,
                                                            dataloaders=dataloaders, phase='train',
                                                            dataset_sizes=dataset_sizes, device=device))

                hist[f'test_ite'].append(test_model_ite(model=model, criterion=criterion, 
                                                            dataloaders=dataloaders, phase='test',
                                                            dataset_sizes=dataset_sizes, device=device))

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)


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

    # Iterate over data.
    for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
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


def test_model_ite(model, criterion, dataloaders, dataset_sizes, phase='test', 
                    running_time=0, half=False, device='cpu', do_print=False):
    """
    Testing function. 
    Print the loss and accuracy after the inference on the testset.
    """
    if do_print: print("\n\n**TESTING**\n")

    sincetime = time.time()

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
        if phase == 'train':
            for idx in range(len(preds)):
                list_y_pred.append(int(preds[idx].cpu()))
                list_y_true.append(int(labels[idx].data.cpu()))
                list_probs.append(probs[idx].cpu())

        else:
            list_y_pred.append(int(preds.cpu()))
            list_y_true.append(int(labels.data.cpu()))
            list_probs.append(probs.cpu())

    #pbar.close()

    test_acc = running_corrects.double() / dataset_sizes[phase]
    test_acc = round(float(test_acc), 4)

    test_loss = log_loss(list_y_true, list_y_pred)

    if do_print: print('\nTest stats -  Loss: {:.4f} Acc: {:.2f}%'.format(test_loss, test_acc*100))            

    if do_print: print("Inference on Testset complete in {:.1f}s\n".format(time.time() - sincetime))

    return [test_loss, test_acc, running_time]