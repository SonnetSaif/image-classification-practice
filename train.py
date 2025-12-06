import os
import torch
import torchvision
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from custom_dataset import CustomDataLoader
from utils import custom_collate
from coco_parser import CocoParser
import json


BASE_PATH = '/kaggle/input/plastic-paper-garbage-bag-synthetic-images/'
NUM_CLASSES = 4
BATCH_SIZE = 16
EPOCHS = 5
LR = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

json_file_path = '/kaggle/input/plastic-paper-garbage-bag-synthetic-images/ImageClassesCombinedWithCOCOAnnotations/coco_instances.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

parser = CocoParser(data)
df = parser.get_final_dataframe()
unique_imgs = df['image_id'].unique()


train_inds, val_inds = train_test_split(range(unique_imgs.shape[0]), test_size=0.1, random_state=42)


train_dl = torch.utils.data.DataLoader(
    CustomDataLoader(df, unique_imgs, train_inds, BASE_PATH),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=custom_collate,
    pin_memory=torch.cuda.is_available()
)

val_dl = torch.utils.data.DataLoader(
    CustomDataLoader(df, unique_imgs, val_inds, BASE_PATH),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate,
    pin_memory=torch.cuda.is_available()
)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

print("📌 Starting Training...\n")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for imgs, targets in train_dl:

        # Skip empty batches (can happen because None images are skipped)
        if len(imgs) == 0:
            continue  

        # Move images & targets to device
        imgs = [img.to(device) for img in imgs]

        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        # Forward pass
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}")


print("\n🎉 Training Complete!")