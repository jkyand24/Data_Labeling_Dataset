import torch
import os
import glob
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from custom_dataset import CustomImageDataset

def is_grayscale(img):
    return img.mode == 'L'

class CachedCustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = glob.glob(os.path.join(image_paths, "*", "*", "*.jpg"))
        self.transform = transform
        self.label_dict = {"dew": 0, "fogsmog": 1, "frost": 2, "glaze": 3, "hail": 4,
                           "lightning": 5, "rain": 6, "rainbow": 7, "rime": 8, "sandstorm": 9,
                           "snow": 10}
        self.cache = {}
        
    def __getitem__(self, index):
        # image, label 가져오기
        
        if index in self.cache:
            image, label = self.cache[index]
            
        else:
            image_path: str = self.image_paths[index]
            image = Image.open(image_path).convert("RGB") # Image.open() -> read an image from disk
            
            if not is_grayscale(image):
                folder_name = image_path.split("\\")[-2]
                label = self.label_dict[folder_name]
                
                self.cache[index] = (image, label) # 메모리에 올림
            
            else:
                print(f"{image_path} 파일은 흑백 이미지입니다.")
                return None, None
            
        # image를 transform
        
        if self.transform:
            image = self.transform(image)
            
        #
            
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
    

if __name__ == "__main__":
    # cached/notcached dataset/dataloader 만들기
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    image_paths = "./data/sample_data_01/"
    
    cached_dataset = CachedCustomImageDataset(image_paths, tf)
    cached_dataloader = DataLoader(cached_dataset, batch_size=64, shuffle=True)
    
    not_cached_dataset = CustomImageDataset(image_paths, tf)
    not_cached_dataloader = DataLoader(not_cached_dataset, batch_size=64, shuffle=True)
    
    # cached/notcached 소요시간 비교
    
    c_start_time = time.time()
    for images, labels in cached_dataloader:
        pass
    print(f"cached class: {time.time() - c_start_time}초 소모")
    
    nc_start_time = time.time()
    for images, labels in not_cached_dataloader:
        pass
    print(f"not cached class: {time.time() - nc_start_time}초 소모")
    
    c_reuse_start_time = time.time()
    for images, labels in cached_dataloader:
        pass
    print(f"cached class reuse: {time.time() - c_reuse_start_time}초 소모")
    
    nc_reuse_start_time = time.time()
    for images, labels in not_cached_dataloader:
        pass
    print(f"not cached class reuse: {time.time() - nc_reuse_start_time}초 소모")
    
    # cached class: 48.53694701194763초 소모 - 처음 사용할 때에는 cache하는 것의 속도상 이점이 별로 없음, 캐시 오버헤드
    # not cached class: 51.79106140136719초 소모
    # cached class reuse: 24.58586311340332초 소모
    # not cached class reuse: 49.815917015075684초 소모