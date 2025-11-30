import pandas as pd

class CocoParser:
    def __init__(self, data):

        annotations = data['annotations']
        images = data['images']
        categories = data['categories']
        
        
        def extract_annotations():
            
            image_ids = []
            category_ids = []
            bbox_x = []
            bbox_y = []
            bbox_width = []
            bbox_height = []
            areas = []
            
            # Iterate over each annotation and extract relevant data
            for annotation in annotations:
                image_ids.append(annotation['image_id'])
                category_ids.append(annotation['category_id'])
                
                # Extract bbox data (x, y, width, height)
                bbox = annotation['bbox']
                bbox_x.append(bbox[0])
                bbox_y.append(bbox[1])
                bbox_width.append(bbox[2])
                bbox_height.append(bbox[3])
                
                # Extract area data
                areas.append(annotation['area'])
            
            
            # Create DataFrame for annotations
            df_annotations = pd.DataFrame({
                'image_id': image_ids,
                'category_id': category_ids,
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_width': bbox_width,
                'bbox_height': bbox_height,
                'area': areas
            })
        
            return df_annotations
        
        
        def extract_images():
        
            licenses = []
            file_names = []
            widths = []
            heights = []
            ids = []
        
            #Iterate over image and extract relevant data
            for image in images:
                licenses.append(image['license'])
                file_names.append(image['file_name'])
                widths.append(image['width'])
                heights.append(image['height'])
                ids.append(image['id'])
        
        
            df_images = pd.DataFrame({
                'image_id': ids,
                'license': licenses,
                'file_name': file_names,
                'width': widths,
                'height': heights
            })
        
            return df_images
        
        
        def extract_categories():
            
            # Create lists to store category data
            category_ids = []
            category_names = []
            supercategories = []
            
            # Iterate over each category data and extract relevant fields
            for category in categories:
                category_ids.append(category['id'])
                category_names.append(category['name'])
                supercategories.append(category['supercategory'])
        
        
            # Create DataFrame for category metadata
            df_categories = pd.DataFrame({
                'category_id': category_ids,
                'category_name': category_names,
                'supercategory': supercategories
            })
        
            return df_categories
        
        
        df_annotations = extract_annotations()
        df_images = extract_images()
        df_categories = extract_categories()
        
        df_combined = pd.merge(df_annotations, df_images, on='image_id')
        df_final = pd.merge(df_combined, df_categories, on='category_id')