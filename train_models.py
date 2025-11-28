import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from data.create_dataset import create_dataset
from models.train_classifier import train_food_classifier
from models.train_calorie_model import train_calorie_regressor

def main():
    print("=" * 60)
    print("AI-Powered Nutrition Recommendation System - Training")
    print("=" * 60)
    
    print("\n[1/3] Creating synthetic food dataset...")
    print("-" * 40)
    create_dataset(samples_per_class=50)
    
    print("\n[2/3] Training food classifier...")
    print("-" * 40)
    classifier, label_encoder, accuracy = train_food_classifier()
    
    print("\n[3/3] Training calorie regressor...")
    print("-" * 40)
    regressor, food_encoder = train_calorie_regressor()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nClassifier Accuracy: {accuracy:.2%}")
    print("\nModels saved in 'models/' directory")
    print("You can now run the Streamlit app!")

if __name__ == "__main__":
    main()
