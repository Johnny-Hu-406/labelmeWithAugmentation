import albumentations as A
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import os
import base64
from tqdm import tqdm
import labelme2coco
import shutil
import random
import yaml

# Load config
def load_config():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def create_directories(config):
    """Create all necessary directories based on enabled features"""
    if config["SplitData"]:
        os.makedirs(config["Output_CoCo_folders"]["labelme_data"]["train_folder"], exist_ok=True)
        os.makedirs(config["Output_CoCo_folders"]["labelme_data"]["val_folder"], exist_ok=True)
    
    if config["Trans2coco"]:
        os.makedirs(config["Output_CoCo_folders"]["coco"]["coco_train"], exist_ok=True)
        os.makedirs(config["Output_CoCo_folders"]["coco"]["coco_val"], exist_ok=True)
        os.makedirs(config["Output_CoCo_folders"]["coco"]["coco_annotations"], exist_ok=True)

def split_dataset(input_folder, train_folder, val_folder, train_ratio=0.8):
    """Split the dataset into train and validation sets"""
    # Get all image files
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Calculate split point
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Copy files to respective folders
    for img_file in train_files:
        json_file = os.path.splitext(img_file)[0] + '.json'
        shutil.copy2(os.path.join(input_folder, img_file), os.path.join(train_folder, img_file))
        shutil.copy2(os.path.join(input_folder, json_file), os.path.join(train_folder, json_file))
    
    for img_file in val_files:
        json_file = os.path.splitext(img_file)[0] + '.json'
        shutil.copy2(os.path.join(input_folder, img_file), os.path.join(val_folder, img_file))
        shutil.copy2(os.path.join(input_folder, json_file), os.path.join(val_folder, json_file))
    
    return len(train_files), len(val_files)

# Your existing helper classes and functions (keep them as is)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Augmentation pipeline
transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, p=0.5, 
                       border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)

# Keep your existing helper functions
def polygon_to_keypoints(polygon):
    return [(float(x), float(y)) for [x, y] in polygon]

def keypoints_to_polygon(keypoints):
    return [[float(x), float(y)] for (x, y) in keypoints]

def update_json_with_image_info(json_data, image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        image_data_b64 = base64.b64encode(image_data).decode("utf-8")
    json_data["imagePath"] = os.path.basename(image_path)
    json_data["imageData"] = image_data_b64
    return json_data

def clip_keypoints_to_image(keypoints, img_width, img_height):
    clipped_keypoints = []
    for x, y in keypoints:
        clipped_x = np.clip(x, 0, img_width - 1)
        clipped_y = np.clip(y, 0, img_height - 1)
        clipped_keypoints.append((clipped_x, clipped_y))
    return clipped_keypoints

def is_polygon_on_boundary(polygon, img_width, img_height, threshold=0.9):
    boundary_points = 0
    total_points = len(polygon)
    for x, y in polygon:
        if (x == 0 or x == img_width - 1 or y == 0 or y == img_height - 1):
            boundary_points += 1
    return (boundary_points / total_points) >= threshold

def process_image_and_json(image_path, json_path, output_path, augmentation_idx):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]
    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data['shapes']
    polygons = [shape['points'] for shape in shapes]
    keypoints = [polygon_to_keypoints(polygon) for polygon in polygons]
    flattened_keypoints = [kp for polygon in keypoints for kp in polygon]
    
    transformed = transform(image=image, keypoints=flattened_keypoints)
    augmented_image = transformed['image']
    augmented_keypoints = transformed['keypoints']

    augmented_polygons = []
    augmented_shapes = []
    for i, shape in enumerate(data['shapes']):
        n_points = len(polygons[i])
        polygon_keypoints = augmented_keypoints[:n_points]
        clipped_keypoints = clip_keypoints_to_image(polygon_keypoints, img_width, img_height)
        
        if len(clipped_keypoints) >= 3:  # 至少需要3個點（首尾會自動封閉形成第4個點）
            if not is_polygon_on_boundary(clipped_keypoints, img_width, img_height):
                augmented_polygons.append(clipped_keypoints)
                shape['points'] = keypoints_to_polygon(clipped_keypoints)
                augmented_shapes.append(shape)
        else:
            print(f"Warning: Skipping invalid polygon with {len(clipped_keypoints)} points in {os.path.basename(image_path)}")
        
        augmented_keypoints = augmented_keypoints[n_points:]

    data['shapes'] = augmented_shapes

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = os.path.join(output_path, f'augmented_{base_name}_{augmentation_idx}.jpg')
    output_json_path = os.path.join(output_path, f'augmented_{base_name}_{augmentation_idx}.json')

    cv2.imwrite(output_image_path, augmented_image)
    data = update_json_with_image_info(data, output_image_path)

    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

    return augmented_image, augmented_polygons

def augment_dataset(input_folder, output_folder, num_augmentations=4):
    """Perform data augmentation on a folder of images and annotations"""
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in tqdm(image_files, desc="Augment_dataset"):
        image_path = os.path.join(input_folder, image_file)
        json_file = os.path.splitext(image_file)[0] + '.json'
        json_path = os.path.join(input_folder, json_file)
        
        if os.path.exists(json_path):
            for i in range(num_augmentations):
                process_image_and_json(image_path, json_path, output_folder, i)
        else:
            # print(f"Warning: No corresponding JSON file found for {image_file}")
            pass

def custom_labelme2coco(config, labelme_folder, export_dir, dataset_type):
    """Convert labelme annotations to COCO format with fixed categories"""
    fixed_categories = config["Fixed_categories"]
    
    # 第一步：將 labelme 資料轉換為初始 COCO 格式
    labelme2coco.convert(labelme_folder, export_dir)
    
    # 讀取生成的初始 JSON 文件
    json_path = os.path.join(export_dir, 'dataset.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 第二步：構建固定的 categories
    existing_categories = {cat['name']: cat['id'] for cat in data['categories']}
    for cat in fixed_categories:
        if cat not in existing_categories:
            existing_categories[cat] = len(existing_categories)
    
    # 按固定順序更新 categories
    data['categories'] = [
        {'id': i, 'name': cat} for i, cat in enumerate(fixed_categories)
    ]
    
    # 構建原始 category_id 到新 category_id 的映射
    id_mapping = {v: fixed_categories.index(k) for k, v in existing_categories.items()}
    
    # 第三步：更新 annotations 的 category_id
    for ann in data['annotations']:
        ann['category_id'] = id_mapping[ann['category_id']]
    
    # 更新圖像名稱的文件路徑
    for image in data['images']:
        image['file_name'] = os.path.basename(image['file_name'])
    
    # 第四步：將處理後的數據寫入新的 JSON 文件
    output_path = os.path.join(export_dir, f'instances_{dataset_type}2017.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # 刪除臨時 JSON 文件
    os.remove(json_path)
    
    # 確認輸出
    print(f"Updated categories for {dataset_type}: {[cat['name'] for cat in data['categories']]}")

def main():
    # Load configuration
    config = load_config()
    
    # Create necessary directories based on enabled features
    create_directories(config)
    
    # Extract paths from config
    input_folder = config["Input_folders"]["input_folder"]
    output_folders = config["Output_CoCo_folders"]
    
    # Initialize processing flags
    split_data = config.get("SplitData", False)
    perform_aug = config.get("PerformAug", False)
    convert_to_coco = config.get("Trans2coco", False)
    
    # Initialize source and destination folders
    source_folder = input_folder
    train_folder = output_folders["labelme_data"]["train_folder"]
    val_folder = output_folders["labelme_data"]["val_folder"]
    coco_train = output_folders["coco"]["coco_train"]
    coco_val = output_folders["coco"]["coco_val"]
    coco_annotations = output_folders["coco"]["coco_annotations"]
    
    # Step 1: Split dataset if enabled
    if split_data:
        # print("Splitting dataset into train and validation sets...")
        train_count, val_count = split_dataset(input_folder, train_folder, val_folder)
        # print(f"Split complete: {train_count} training samples, {val_count} validation samples")
        # Update source folders for augmentation
        source_train = train_folder
        source_val = val_folder
        print("Split data complete!")
    else:
        # print("Skipping dataset split (SplitData is False)")
        # If not splitting, use input folder directly
        source_train = source_val = input_folder
    
    # Step 2: Perform augmentation if enabled
    if perform_aug:
        # print("Starting data augmentation...")
        if split_data:
            # print("Augmenting training set...")
            augment_dataset(source_train, coco_train)
            # print("Augmenting validation set...")
            # augment_dataset(source_val, coco_val)
            shutil.copytree(source_val, coco_val, dirs_exist_ok = True)
    else:
        print("Skipping data augmentation (PerformAug is False)")
        pass
    
    # Step 3: Convert to COCO format if enabled
    if perform_aug and split_data and  convert_to_coco:
        custom_labelme2coco(config, coco_train, coco_annotations, 'train')
        custom_labelme2coco(config, coco_val, coco_annotations, 'val')
    elif convert_to_coco:
        input_folder = 'input_data/FongSiang'
        val_files = os.listdir(input_folder)  
        for img_file in val_files:
            # json_file = os.path.splitext(img_file)[0] + '.json'
            shutil.copy2(os.path.join(input_folder, img_file), os.path.join(coco_val, img_file))
            # shutil.copy2(os.path.join(input_folder, json_file), os.path.join(val_folder, json_file))
        custom_labelme2coco(config, coco_val, coco_annotations, 'val')
    
    print("All Processing complete!")

if __name__ == "__main__":
    main()