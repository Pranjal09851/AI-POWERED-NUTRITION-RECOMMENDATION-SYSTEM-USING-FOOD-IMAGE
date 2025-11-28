import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FOOD_CATEGORIES = [
    "apple", "banana", "burger", "pizza", "salad",
    "sandwich", "pasta", "rice", "chicken", "fish",
    "bread", "egg", "soup", "steak", "sushi"
]

IMAGE_SIZE = (128, 128)

NUTRITION_DATA = {
    "apple": {"calories": 95, "protein": 0.5, "carbs": 25, "fat": 0.3, "fiber": 4.4},
    "banana": {"calories": 105, "protein": 1.3, "carbs": 27, "fat": 0.4, "fiber": 3.1},
    "burger": {"calories": 540, "protein": 25, "carbs": 40, "fat": 29, "fiber": 2},
    "pizza": {"calories": 285, "protein": 12, "carbs": 36, "fat": 10, "fiber": 2.5},
    "salad": {"calories": 150, "protein": 5, "carbs": 12, "fat": 10, "fiber": 4},
    "sandwich": {"calories": 350, "protein": 15, "carbs": 35, "fat": 16, "fiber": 3},
    "pasta": {"calories": 380, "protein": 14, "carbs": 75, "fat": 2, "fiber": 3},
    "rice": {"calories": 205, "protein": 4.3, "carbs": 45, "fat": 0.4, "fiber": 0.6},
    "chicken": {"calories": 335, "protein": 38, "carbs": 0, "fat": 20, "fiber": 0},
    "fish": {"calories": 206, "protein": 22, "carbs": 0, "fat": 12, "fiber": 0},
    "bread": {"calories": 79, "protein": 2.7, "carbs": 15, "fat": 1, "fiber": 0.6},
    "egg": {"calories": 155, "protein": 13, "carbs": 1.1, "fat": 11, "fiber": 0},
    "soup": {"calories": 120, "protein": 6, "carbs": 15, "fat": 4, "fiber": 2},
    "steak": {"calories": 679, "protein": 62, "carbs": 0, "fat": 48, "fiber": 0},
    "sushi": {"calories": 200, "protein": 9, "carbs": 38, "fat": 1, "fiber": 1}
}

DIETARY_GOALS = {
    "weight_loss": {
        "daily_calories": 1500,
        "protein_ratio": 0.35,
        "carbs_ratio": 0.35,
        "fat_ratio": 0.30,
        "description": "Focus on high protein, moderate carbs, and healthy fats"
    },
    "maintenance": {
        "daily_calories": 2000,
        "protein_ratio": 0.25,
        "carbs_ratio": 0.50,
        "fat_ratio": 0.25,
        "description": "Balanced nutrition to maintain current weight"
    },
    "muscle_gain": {
        "daily_calories": 2500,
        "protein_ratio": 0.40,
        "carbs_ratio": 0.40,
        "fat_ratio": 0.20,
        "description": "High protein and carbs for muscle building"
    }
}
