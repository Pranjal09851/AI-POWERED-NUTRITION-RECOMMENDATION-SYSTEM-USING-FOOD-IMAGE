import os
import sys
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, PROCESSED_DATA_DIR

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    return img_array

def extract_features(img_array):
    gray = rgb2gray(img_array)
    
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    
    color_hist = []
    for channel in range(3):
        hist, _ = np.histogram(img_array[:, :, channel], bins=32, range=(0, 1))
        color_hist.extend(hist / hist.sum())
    
    mean_color = img_array.mean(axis=(0, 1))
    std_color = img_array.std(axis=(0, 1))
    
    features = np.concatenate([
        hog_features,
        np.array(color_hist),
        mean_color,
        std_color
    ])
    
    return features

def preprocess_dataset(df):
    features_list = []
    labels = []
    
    for idx, row in df.iterrows():
        try:
            img_array = load_and_preprocess_image(row['image_path'])
            features = extract_features(img_array)
            features_list.append(features)
            labels.append(row['food_type'])
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
    
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y

def preprocess_single_image(image):
    if isinstance(image, str):
        img_array = load_and_preprocess_image(image)
    elif isinstance(image, Image.Image):
        img = image.convert('RGB').resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
    else:
        img_array = image
        if img_array.max() > 1:
            img_array = img_array / 255.0
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        if img_array.shape[:2] != IMAGE_SIZE:
            img_array = resize(img_array, IMAGE_SIZE)
    
    features = extract_features(img_array)
    return features.reshape(1, -1)

if __name__ == "__main__":
    csv_path = os.path.join(PROCESSED_DATA_DIR, "nutrition_dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X, y = preprocess_dataset(df)
        print(f"Feature shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        np.save(os.path.join(PROCESSED_DATA_DIR, "features.npy"), X)
        np.save(os.path.join(PROCESSED_DATA_DIR, "labels.npy"), y)
        print("Features saved successfully!")
    else:
        print("Dataset not found. Run create_dataset.py first.")
