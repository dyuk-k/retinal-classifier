import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from src.preprocessing import apply_clahe
from PIL import Image

class RetinalDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        
        self.base_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        
        self.aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(30),transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
        img = apply_clahe(img_path)
        img = Image.fromarray(img)
        
        if self.augment:
            img = self.aug_transforms(img)
        else:
            img = self.base_transforms(img)
            
        return img, torch.tensor(row['diagnosis'], dtype=torch.long)