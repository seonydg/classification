import torch
import torch.nn.functional as F
from utils import calculate_accuracy

def train(model, dataloader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for image, label in dataloader['train']:
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        pred, _ = model(image)
        loss = criterion(pred, label)
        acc = calculate_accuracy(pred, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader['train']), epoch_acc / len(dataloader['train'])


def evaluate(model, dataloader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for image, label in dataloader['val']:
            image = image.to(device)
            label = label.to(device)

            pred, _ = model(image)
            loss = criterion(pred, label)
            acc = calculate_accuracy(pred, label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader['val']), epoch_acc / len(dataloader['val'])


def predict(model, dataloader, device):
    images = []
    labels = []
    probs = []

    model.eval()
    with torch.no_grad():
        for image, label in dataloader['test']:
            image = image.to(device)
            pred, _ = model(image)
            prob = F.softmax(pred, dim=1)
            top_pred = prob.argmax(1, keepdim=True)

            images.append(image.cpu())
            labels.append(label.cpu())
            probs.append(prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    return images, labels, probs