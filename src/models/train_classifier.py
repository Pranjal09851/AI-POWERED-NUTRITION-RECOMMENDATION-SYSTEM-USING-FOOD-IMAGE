import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR, MODELS_DIR, FOOD_CATEGORIES
from data.preprocess import preprocess_dataset

def train_food_classifier():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    csv_path = os.path.join(PROCESSED_DATA_DIR, "nutrition_dataset.csv")
    df = pd.read_csv(csv_path)
    
    print("Extracting features from images...")
    X, y = preprocess_dataset(df)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print("Training RandomForest classifier...")
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nClassifier Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_
    ))
    
    classifier_path = os.path.join(MODELS_DIR, "food_classifier.joblib")
    encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    
    joblib.dump(classifier, classifier_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\nClassifier saved to {classifier_path}")
    print(f"Label encoder saved to {encoder_path}")
    
    return classifier, label_encoder, accuracy

if __name__ == "__main__":
    train_food_classifier()
