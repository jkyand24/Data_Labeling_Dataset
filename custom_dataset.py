import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Data Set 반환하는 클래스 구현

def is_grayscale(img):
    return img.mode == 'L'

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}
    
    def __getitem__(self, index):
        image_path: str = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if not is_grayscale(image):
            if self.transform:
                image = self.transform(image)
            
            folder_name = image_path.split("\\")[2]
            label = self.label_dict[folder_name]
                
            return image, label
        
        else:
            print("흑백 이미지 >>", image_path)
    
    def __len__(self):
        return len(self.image_paths)
    
#

if __name__ == "__main__": # if 모듈 임포트가 아니라 인터프리터에서 직접 실행:
    # Data Set

    image_paths = "./data/sample_data_01/"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomImageDataset(image_paths, transform=transform)

    # Data Loader
    
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for images, labels in data_loader: # batch_size개씩 묶여있음
        print(f"Data and Label: {images}, {labels}")
        exit()