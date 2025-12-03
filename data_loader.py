import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms

class CustomDataLoader(torch.utils.data.Dataset):
    def __init__(self, df, unique_imgs, indices, base_path):
        self.df = df
        self.unique_imgs = unique_imgs
        self.indices = indices
        self.base_path = base_path

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = f"{self.image_dir}/{row['file_name']}"
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        bbox = torch.tensor([row['x1'], row['y1'], row['x2'], row['y2']], dtype=torch.float32)
        label = torch.tensor(row['category_id'], dtype=torch.long)
        
        return image, bbox, label
