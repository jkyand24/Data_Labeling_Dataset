import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, json_path, transforms=None):
        self.transforms = transforms
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __getitem__(self, index):
        print("\nself.data:\n", self.data)
        print("")
        
        filename = self.data[index]['filename']
        img_path = os.path.join("이미지 폴더", filename)
        
        bboxes = self.data[index]['ann']['bboxes']
        labels = self.data[index]['ann']['labels']
        
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return img_path, {'bboxes': bboxes, 'labels': labels}
    
    def __len__(self):
        return len(self.data)
    
dataset = CustomDataset("./data/test.json", transforms=None)

for i in dataset:
    print(i) # __getitem__()의 return 값도 나옴