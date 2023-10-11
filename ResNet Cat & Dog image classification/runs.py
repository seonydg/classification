import torch
from utils import calculate_accuracy


def train(model, dataloaders, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.train()
    for image, label in dataloaders:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred = model(image)
        loss = criterion(pred, label)

        acc_1, acc_5 = calculate_accuracy(pred=pred, label=label)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()
    
    epoch_loss /= len(dataloaders)
    epoch_acc_1 /= len(dataloaders)
    epoch_acc_5 /= len(dataloaders)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def evaluate(model, dataloaders, criterion, device):
    epoch_loss = 0
    epoch_acc_1 = 0
    epoch_acc_5 = 0

    model.eval()
    with torch.no_grad():
        for image, label in dataloaders:
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            loss = criterion(pred, label)

            acc_1, acc_5 = calculate_accuracy(pred=pred, label=label)
            epoch_loss += loss.item()
            epoch_acc_1 += acc_1.item()
            epoch_acc_5 += acc_5.item()
        
    epoch_loss /= len(dataloaders)
    epoch_acc_1 /= len(dataloaders)
    epoch_acc_5 /= len(dataloaders)

    return epoch_loss, epoch_acc_1, epoch_acc_5