import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, NUTRITION_DATA, FOOD_CATEGORIES

def create_nutrition_features(df):
    label_encoder = LabelEncoder()
    food_encoded = label_encoder.fit_transform(df['food_type'])
    
    features = []
    for idx, row in df.iterrows():
        food_type = row['food_type']
        base_nutrition = NUTRITION_DATA[food_type]
        
        feature_vector = [
            food_encoded[idx],
            base_nutrition['protein'],
            base_nutrition['carbs'],
            base_nutrition['fat'],
            base_nutrition['fiber'],
            base_nutrition['protein'] * 4,
            base_nutrition['carbs'] * 4,
            base_nutrition['fat'] * 9,
        ]
        features.append(feature_vector)
    
    return np.array(features), label_encoder

def train_calorie_regressor():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    csv_path = os.path.join(PROCESSED_DATA_DIR, "nutrition_dataset.csv")
    df = pd.read_csv(csv_path)
    
    print("Creating nutrition features...")
    X, food_encoder = create_nutrition_features(df)
    y = df['calories'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training RandomForest regressor for calorie prediction...")
    regressor = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nCalorie Prediction Metrics:")
    print(f"  MAE: {mae:.2f} calories")
    print(f"  RMSE: {rmse:.2f} calories")
    print(f"  R2 Score: {r2:.4f}")
    
    regressor_path = os.path.join(MODELS_DIR, "calorie_regressor.joblib")
    food_encoder_path = os.path.join(MODELS_DIR, "food_encoder.joblib")
    
    joblib.dump(regressor, regressor_path)
    joblib.dump(food_encoder, food_encoder_path)
    
    print(f"\nRegressor saved to {regressor_path}")
    print(f"Food encoder saved to {food_encoder_path}")
    
    return regressor, food_encoder

if __name__ == "__main__":
    train_calorie_regressor()
