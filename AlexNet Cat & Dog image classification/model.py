import torch
import torch.nn as nn
import time
from tqdm.notebook import tqdm


class AlexNet(nn.Module):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
                                    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2),

                                    nn.Conv2d(64, 192, kernel_size=5, padding=2),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2),

                                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(384, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),

                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
                                    nn.Dropout(),
                                    nn.Linear(256 * 6 * 6, 4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(),
                                    nn.Linear(4096, 512),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(512, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def train_one_epoch(model, dataloader, criterion, optimizer, num_epochs, device):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase  in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_acc = 0

            for images, labels in tqdm(dataloader[phase]):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    out = model(images)
                    _, preds = torch.max(out, 1)
                    loss = criterion(out, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                    running_loss += loss.item() * images.size(0)
                    running_acc += torch.sum(preds == labels.data)
            
            running_losses = running_loss / len(dataloader[phase].dataset)
            running_accs = running_acc.double() / len(dataloader[phase].dataset)

            print(f'{phase} Loss: {running_losses} Acc: {running_accs}')
        
    time_elapsed = time.time() - since
    print(f'training complete in {time_elapsed // 60}m {time_elapsed % 60}')
    
    return model

