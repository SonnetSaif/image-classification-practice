import json
from coco_parser import CocoParser

# Load the JSON file
json_file_path = '/kaggle/input/plastic-paper-garbage-bag-synthetic-images/ImageClassesCombinedWithCOCOAnnotations/coco_instances.json'
with open(json_file_path, 'r') as file:
    data = json.load(file)

parser = CocoParser(data)
df = parser.get_final_dataframe()
print(df.head())
unique_images = df['image_id'].unique()