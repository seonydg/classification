import torchvision.transforms as transforms
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