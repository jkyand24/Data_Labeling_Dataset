from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class AlbumentationsDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        file_path = self.file_paths[index]
        
        label = self.labels[index]
        
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
albumentations_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(),
    A.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225],
    ),
    ToTensorV2
])    

albumentations_dataset = AlbumentationsDataset(
    file_paths=['./data/sample_data_01/train/dew/2208.jpg', './data/sample_data_01/train/fogsmog/4075.jpg', './data/sample_data_01/train/frost/3600.jpg'],
    labels = [0, 1, 2],
    transform=albumentations_transform,
)    