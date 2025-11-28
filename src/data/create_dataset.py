import os
import sys
import numpy as np
from PIL import Image
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FOOD_CATEGORIES, NUTRITION_DATA, RAW_DATA_DIR, PROCESSED_DATA_DIR, IMAGE_SIZE

def generate_synthetic_food_image(food_type, seed):
    np.random.seed(seed)
    
    color_map = {
        "apple": (200, 50, 50),
        "banana": (255, 230, 50),
        "burger": (150, 100, 60),
        "pizza": (220, 160, 80),
        "salad": (80, 180, 80),
        "sandwich": (200, 180, 140),
        "pasta": (240, 220, 180),
        "rice": (250, 250, 240),
        "chicken": (210, 170, 120),
        "fish": (180, 200, 210),
        "bread": (210, 180, 140),
        "egg": (255, 240, 200),
        "soup": (180, 100, 80),
        "steak": (130, 80, 70),
        "sushi": (220, 200, 180)
    }
    
    base_color = np.array(color_map.get(food_type, (128, 128, 128)))
    
    img = np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    
    noise = np.random.randint(-30, 30, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    for i in range(3):
        channel = np.full((IMAGE_SIZE[0], IMAGE_SIZE[1]), base_color[i], dtype=np.int32)
        channel = channel + noise[:, :, i]
        channel = np.clip(channel, 0, 255)
        img[:, :, i] = channel.astype(np.uint8)
    
    center_x, center_y = IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2
    for i in range(IMAGE_SIZE[0]):
        for j in range(IMAGE_SIZE[1]):
            dist = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
            if dist < IMAGE_SIZE[0] // 3:
                img[i, j] = np.clip(img[i, j].astype(np.int32) + 20, 0, 255).astype(np.uint8)
    
    texture = np.random.randint(-15, 15, (IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    img = np.clip(img.astype(np.int32) + texture, 0, 255).astype(np.uint8)
    
    return Image.fromarray(img, 'RGB')

def create_dataset(samples_per_class=50):
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    dataset_info = []
    
    for food_type in FOOD_CATEGORIES:
        food_dir = os.path.join(RAW_DATA_DIR, food_type)
        os.makedirs(food_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            seed = hash(f"{food_type}_{i}") % (2**32)
            img = generate_synthetic_food_image(food_type, seed)
            
            img_path = os.path.join(food_dir, f"{food_type}_{i:03d}.png")
            img.save(img_path)
            
            nutrition = NUTRITION_DATA[food_type]
            variation = np.random.uniform(0.85, 1.15)
            
            dataset_info.append({
                "image_path": img_path,
                "food_type": food_type,
                "calories": nutrition["calories"] * variation,
                "protein": nutrition["protein"] * variation,
                "carbs": nutrition["carbs"] * variation,
                "fat": nutrition["fat"] * variation,
                "fiber": nutrition["fiber"] * variation
            })
    
    df = pd.DataFrame(dataset_info)
    csv_path = os.path.join(PROCESSED_DATA_DIR, "nutrition_dataset.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"Created {len(dataset_info)} images across {len(FOOD_CATEGORIES)} categories")
    print(f"Dataset saved to {csv_path}")
    
    return df

if __name__ == "__main__":
    create_dataset(samples_per_class=50)
