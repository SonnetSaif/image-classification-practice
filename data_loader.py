import os
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
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_id = self.unique_imgs[self.indices[idx]]

        if img_id > 5000:
            print(f"Skipping image_id: {img_id}")
            return None, None
        
        print(f"Processing image_id: {img_id}")

        img_data = self.df[self.df['image_id'] == img_id]

        if img_data.empty:
            print(f"No data found for image_id: {img_id}")
            return None, None
        
        boxes = img_data[['x1', 'y1', 'x2', 'y2']].values.astype('float')
        category_name  = img_data['category_name'].values[0]
        file_name = img_data['file_name'].values[0]

        category_name = category_name.capitalize()

        file_path = os.path.join(self.base_path, 'Bag Classes', 'Bag Classes', category_name + ' Bag Images', file_name)
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None, None
        
        # Load the image
        img = Image.open(file_path).convert('RGB')

        # Create labels for bounding boxes
        labels = torch.ones(boxes.shape[0], dtype=torch.int64)

        # Prepare the target dictionary with boxes and labels
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = labels

        # Convert the image to a tensor
        img = transforms.ToTensor()(img)

        return img, target