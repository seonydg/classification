import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import Dataset

class build_trasnforms():
    def __init__(self, image_size, mean, std):
        self.transforms = {
                        'train':transforms.Compose([
                                                    transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)
                        ]),
                        'val':transforms.Compose([
                                                    transforms.Resize(image_size),
                                                    transforms.CenterCrop(image_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std)
                        ])
        }
    
    def __call__(self, image, phase):
        return self.transforms[phase](image)


class MyDataset(Dataset):
    def __init__(self, file_list, transforms=None, phase='train'):
        self.file_list = file_list
        self.transforms = transforms
        self.phase = phase
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        image_path = self.file_list[index]
        image = Image.open(image_path)
        image_transform = self.transforms(image, self.phase)
        label = image_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        else:
            label = 0
        
        return image_transform, label
    

def calculate_accuracy(pred, label, k=2):
    with torch.no_grad():
        batch_size = label.shape[0]
        _, top_pred = pred.topk(k, 1)
        top_pred = top_pred.t()

        correct = top_pred.eq(label.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    
    return acc_1, acc_k


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_min = int(elapsed_time/60)
    elapsed_sec = int(elapsed_time - (elapsed_min*60))

    return elapsed_min, elapsed_sec


