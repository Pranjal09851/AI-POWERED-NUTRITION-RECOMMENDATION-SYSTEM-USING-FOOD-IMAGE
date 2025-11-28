import os
import sys
import numpy as np
from PIL import Image
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODELS_DIR, NUTRITION_DATA, DIETARY_GOALS
from data.preprocess import preprocess_single_image

class NutritionPredictor:
    def __init__(self):
        self.classifier = None
        self.label_encoder = None
        self.calorie_regressor = None
        self.food_encoder = None
        self._load_models()
    
    def _load_models(self):
        classifier_path = os.path.join(MODELS_DIR, "food_classifier.joblib")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
        regressor_path = os.path.join(MODELS_DIR, "calorie_regressor.joblib")
        food_encoder_path = os.path.join(MODELS_DIR, "food_encoder.joblib")
        
        if os.path.exists(classifier_path):
            self.classifier = joblib.load(classifier_path)
            self.label_encoder = joblib.load(encoder_path)
            print("Food classifier loaded successfully")
        else:
            print("Warning: Food classifier not found. Run training first.")
        
        if os.path.exists(regressor_path):
            self.calorie_regressor = joblib.load(regressor_path)
            self.food_encoder = joblib.load(food_encoder_path)
            print("Calorie regressor loaded successfully")
        else:
            print("Warning: Calorie regressor not found. Run training first.")
    
    def predict_food(self, image):
        if self.classifier is None:
            return None, 0.0
        
        features = preprocess_single_image(image)
        
        prediction = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        food_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction]
        
        return food_type, confidence
    
    def predict_calories(self, food_type):
        if food_type not in NUTRITION_DATA:
            return None
        
        base_nutrition = NUTRITION_DATA[food_type]
        
        if self.calorie_regressor is not None and self.food_encoder is not None:
            try:
                food_encoded = self.food_encoder.transform([food_type])[0]
                features = np.array([[
                    food_encoded,
                    base_nutrition['protein'],
                    base_nutrition['carbs'],
                    base_nutrition['fat'],
                    base_nutrition['fiber'],
                    base_nutrition['protein'] * 4,
                    base_nutrition['carbs'] * 4,
                    base_nutrition['fat'] * 9,
                ]])
                predicted_calories = self.calorie_regressor.predict(features)[0]
                return predicted_calories
            except Exception:
                pass
        
        return base_nutrition['calories']
    
    def get_nutrition_info(self, food_type):
        if food_type not in NUTRITION_DATA:
            return None
        
        nutrition = NUTRITION_DATA[food_type].copy()
        nutrition['calories'] = self.predict_calories(food_type)
        return nutrition
    
    def get_dietary_suggestions(self, food_type, goal, consumed_today=None):
        if consumed_today is None:
            consumed_today = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        
        nutrition = self.get_nutrition_info(food_type)
        if nutrition is None:
            return None
        
        goal_info = DIETARY_GOALS.get(goal, DIETARY_GOALS['maintenance'])
        
        daily_target = goal_info['daily_calories']
        protein_target = daily_target * goal_info['protein_ratio'] / 4
        carbs_target = daily_target * goal_info['carbs_ratio'] / 4
        fat_target = daily_target * goal_info['fat_ratio'] / 9
        
        after_meal = {
            'calories': consumed_today['calories'] + nutrition['calories'],
            'protein': consumed_today['protein'] + nutrition['protein'],
            'carbs': consumed_today['carbs'] + nutrition['carbs'],
            'fat': consumed_today['fat'] + nutrition['fat']
        }
        
        remaining = {
            'calories': max(0, daily_target - after_meal['calories']),
            'protein': max(0, protein_target - after_meal['protein']),
            'carbs': max(0, carbs_target - after_meal['carbs']),
            'fat': max(0, fat_target - after_meal['fat'])
        }
        
        suggestions = []
        
        if after_meal['calories'] > daily_target:
            excess = after_meal['calories'] - daily_target
            suggestions.append(f"This meal would put you {excess:.0f} calories over your daily target.")
            suggestions.append("Consider a lighter option or smaller portion.")
        else:
            suggestions.append(f"You'll have {remaining['calories']:.0f} calories remaining for the day.")
        
        if goal == 'weight_loss':
            if nutrition['protein'] >= 20:
                suggestions.append("Good protein content for satiety!")
            if nutrition['fiber'] >= 3:
                suggestions.append("High fiber helps keep you full longer.")
            if nutrition['calories'] > 400:
                suggestions.append("Consider a lower calorie alternative for faster results.")
        
        elif goal == 'muscle_gain':
            if nutrition['protein'] >= 25:
                suggestions.append("Excellent protein content for muscle building!")
            elif nutrition['protein'] < 15:
                suggestions.append("Consider adding a protein source to this meal.")
            if nutrition['carbs'] >= 30:
                suggestions.append("Good carbs for energy and recovery.")
        
        elif goal == 'maintenance':
            if nutrition['calories'] < 500:
                suggestions.append("Well-balanced meal for maintenance.")
            suggestions.append("Try to maintain variety in your diet.")
        
        return {
            'food_type': food_type,
            'nutrition': nutrition,
            'goal': goal,
            'goal_description': goal_info['description'],
            'daily_target': daily_target,
            'after_meal': after_meal,
            'remaining': remaining,
            'suggestions': suggestions
        }
    
    def predict(self, image, goal='maintenance', consumed_today=None):
        food_type, confidence = self.predict_food(image)
        
        if food_type is None:
            return {
                'success': False,
                'error': 'Could not classify food image'
            }
        
        suggestions = self.get_dietary_suggestions(food_type, goal, consumed_today)
        
        return {
            'success': True,
            'food_type': food_type,
            'confidence': confidence,
            'suggestions': suggestions
        }

if __name__ == "__main__":
    predictor = NutritionPredictor()
    
    result = predictor.get_dietary_suggestions('pizza', 'weight_loss')
    if result:
        print(f"\nFood: {result['food_type']}")
        print(f"Calories: {result['nutrition']['calories']:.0f}")
        print(f"\nSuggestions for {result['goal']}:")
        for suggestion in result['suggestions']:
            print(f"  - {suggestion}")
