import pandas as pd

class CocoParser:
    def __init__(self, data):
        self.annotations = data['annotations']
        self.images = data['images']
        self.categories = data['categories']
        
    def extract_annotations(self):
        image_ids = []
        category_ids = []
        bbox_x = []
        bbox_y = []
        bbox_width = []
        bbox_height = []
        areas = []
        
        for annotation in self.annotations:
            image_ids.append(annotation['image_id'])
            category_ids.append(annotation['category_id'])
            
            bbox = annotation['bbox']
            bbox_x.append(bbox[0])
            bbox_y.append(bbox[1])
            bbox_width.append(bbox[2])
            bbox_height.append(bbox[3])
            
            areas.append(annotation['area'])
        
        return pd.DataFrame({
            'image_id': image_ids,
            'category_id': category_ids,
            'bbox_x': bbox_x,
            'bbox_y': bbox_y,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'area': areas
        })
    
    def extract_images(self):
        licenses = []
        file_names = []
        widths = []
        heights = []
        ids = []
        
        for image in self.images:
            licenses.append(image['license'])
            file_names.append(image['file_name'])
            widths.append(image['width'])
            heights.append(image['height'])
            ids.append(image['id'])
        
        return pd.DataFrame({
            'image_id': ids,
            'license': licenses,
            'file_name': file_names,
            'width': widths,
            'height': heights
        })
    
    def extract_categories(self):
        category_ids = []
        category_names = []
        supercategories = []
        
        for category in self.categories:
            category_ids.append(category['id'])
            category_names.append(category['name'])
            supercategories.append(category['supercategory'])
        
        return pd.DataFrame({
            'category_id': category_ids,
            'category_name': category_names,
            'supercategory': supercategories
        })

    def get_final_dataframe(self):
        df_annotations = self.extract_annotations()
        df_images = self.extract_images()
        df_categories = self.extract_categories()

        df_combined = pd.merge(df_annotations, df_images, on='image_id')
        df_final = pd.merge(df_combined, df_categories, on='category_id')

        df_final = df_final.rename(columns={'bbox_x': 'x1', 'bbox_y': 'y1'})
        df_final = df_final[df_final['image_id'] <= 5000]

        df_final['x2'] = df_final['x1'] + df_final['bbox_width']
        df_final['y2'] = df_final['y1'] + df_final['bbox_height']

        return df_final