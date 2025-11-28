import streamlit as st
import os
import sys
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FOOD_CATEGORIES, NUTRITION_DATA, DIETARY_GOALS
from pipeline.predict import NutritionPredictor

st.set_page_config(
    page_title="AI Nutrition Advisor",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 50%, #81C784 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(46, 125, 50, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .food-detected {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #4CAF50;
    }
    
    .food-name {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E7D32;
        margin: 0;
    }
    
    .confidence-badge {
        display: inline-block;
        background: #4CAF50;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    .suggestion-card {
        background: #FFF8E1;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid #FFC107;
    }
    
    .suggestion-card.tip {
        background: #E3F2FD;
        border-left-color: #2196F3;
    }
    
    .suggestion-card.warning {
        background: #FFEBEE;
        border-left-color: #F44336;
    }
    
    .suggestion-card.success {
        background: #E8F5E9;
        border-left-color: #4CAF50;
    }
    
    .progress-container {
        background: #E8F5E9;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .food-category-chip {
        display: inline-block;
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9rem;
        border: 1px solid #A5D6A7;
        transition: all 0.2s ease;
    }
    
    .food-category-chip:hover {
        background: linear-gradient(135deg, #C8E6C9 0%, #A5D6A7 100%);
        transform: scale(1.05);
    }
    
    .upload-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 1rem 0;
    }
    
    .sidebar-section {
        background: rgba(255,255,255,0.7);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    .goal-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #ffffff 0%, #f5f5f5 100%);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 3px solid #4CAF50;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%);
        border-radius: 10px;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #E8F5E9 0%, #C8E6C9 100%);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #E0E0E0;
        margin-top: 3rem;
    }
</style>
"""

FOOD_EMOJIS = {
    "apple": "🍎", "banana": "🍌", "burger": "🍔", "pizza": "🍕",
    "salad": "🥗", "sandwich": "🥪", "pasta": "🍝", "rice": "🍚",
    "chicken": "🍗", "fish": "🐟", "bread": "🍞", "egg": "🥚",
    "soup": "🍲", "steak": "🥩", "sushi": "🍣"
}

GOAL_EMOJIS = {
    "weight_loss": "🎯",
    "maintenance": "⚖️",
    "muscle_gain": "💪"
}

@st.cache_resource
def load_predictor():
    return NutritionPredictor()

def render_metric_card(icon, label, value, color="#4CAF50"):
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <p class="metric-value" style="color: {color};">{value}</p>
            <p class="metric-label">{label}</p>
        </div>
    """, unsafe_allow_html=True)

def render_suggestion(text, suggestion_type="default"):
    icon_map = {"tip": "💡", "warning": "⚠️", "success": "✅", "default": "📌"}
    icon = icon_map.get(suggestion_type, "📌")
    st.markdown(f"""
        <div class="suggestion-card {suggestion_type}">
            {icon} {text}
        </div>
    """, unsafe_allow_html=True)

def get_suggestion_type(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ["over", "excess", "reached", "consider a lower", "lighter"]):
        return "warning"
    elif any(word in text_lower for word in ["good", "excellent", "great", "high fiber"]):
        return "success"
    elif any(word in text_lower for word in ["consider", "try", "adding"]):
        return "tip"
    return "default"

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🥗 AI Nutrition Advisor</h1>
            <p>Upload a food image to get instant calorie estimates and personalized dietary recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🎯 Your Goals")
        
        dietary_goal = st.selectbox(
            "Select Dietary Goal",
            options=list(DIETARY_GOALS.keys()),
            format_func=lambda x: f"{GOAL_EMOJIS.get(x, '')} {x.replace('_', ' ').title()}"
        )
        
        goal_info = DIETARY_GOALS[dietary_goal]
        st.markdown(f"""
            <div class="goal-card">
                <strong>{GOAL_EMOJIS.get(dietary_goal, '')} {dietary_goal.replace('_', ' ').title()}</strong><br>
                <small>{goal_info['description']}</small><br>
                <strong style="color: #2E7D32;">{goal_info['daily_calories']} cal/day</strong>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Today's Intake")
        
        if 'calories_consumed' not in st.session_state:
            st.session_state.calories_consumed = 0
        if 'protein_consumed' not in st.session_state:
            st.session_state.protein_consumed = 0.0
        if 'carbs_consumed' not in st.session_state:
            st.session_state.carbs_consumed = 0.0
        if 'fat_consumed' not in st.session_state:
            st.session_state.fat_consumed = 0.0
        
        calories_consumed = st.number_input(
            "🔥 Calories", min_value=0, 
            value=st.session_state.calories_consumed, step=50,
            key="cal_input"
        )
        protein_consumed = st.number_input(
            "🥩 Protein (g)", min_value=0.0, 
            value=st.session_state.protein_consumed, step=5.0,
            key="prot_input"
        )
        carbs_consumed = st.number_input(
            "🍞 Carbs (g)", min_value=0.0, 
            value=st.session_state.carbs_consumed, step=5.0,
            key="carb_input"
        )
        fat_consumed = st.number_input(
            "🧈 Fat (g)", min_value=0.0, 
            value=st.session_state.fat_consumed, step=5.0,
            key="fat_input"
        )
        
        st.session_state.calories_consumed = calories_consumed
        st.session_state.protein_consumed = protein_consumed
        st.session_state.carbs_consumed = carbs_consumed
        st.session_state.fat_consumed = fat_consumed
        
        consumed_today = {
            'calories': calories_consumed,
            'protein': protein_consumed,
            'carbs': carbs_consumed,
            'fat': fat_consumed
        }
        
        current_progress = min(calories_consumed / goal_info['daily_calories'], 1.0)
        st.markdown("**Daily Progress**")
        st.progress(current_progress)
        remaining = max(0, goal_info['daily_calories'] - calories_consumed)
        st.caption(f"{remaining:.0f} calories remaining")
    
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Upload Food Image")
        
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="", use_container_width=True)
        else:
            st.markdown("""
                <div class="upload-section">
                    <div style="font-size: 4rem; margin-bottom: 1rem;">📷</div>
                    <h4 style="color: #2E7D32; margin: 0;">Drop your food photo here</h4>
                    <p style="color: #666; margin: 0.5rem 0;">or click to browse files</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner("🔍 Analyzing your food..."):
                try:
                    predictor = load_predictor()
                    result = predictor.predict(image, dietary_goal, consumed_today)
                    
                    if result['success']:
                        food_type = result['food_type']
                        confidence = result['confidence']
                        suggestions = result['suggestions']
                        food_emoji = FOOD_EMOJIS.get(food_type, "🍽️")
                        
                        st.markdown(f"""
                            <div class="food-detected">
                                <div style="font-size: 3rem;">{food_emoji}</div>
                                <p class="food-name">{food_type.title()}</p>
                                <span class="confidence-badge">
                                    {confidence:.0%} confidence
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🍽️ Nutritional Breakdown")
                        nutrition = suggestions['nutrition']
                        
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("🔥 Calories", f"{nutrition['calories']:.0f}")
                        with m2:
                            st.metric("🥩 Protein", f"{nutrition['protein']:.1f}g")
                        with m3:
                            st.metric("🍞 Carbs", f"{nutrition['carbs']:.1f}g")
                        
                        m4, m5 = st.columns(2)
                        with m4:
                            st.metric("🧈 Fat", f"{nutrition['fat']:.1f}g")
                        with m5:
                            st.metric("🌾 Fiber", f"{nutrition['fiber']:.1f}g")
                        
                        st.markdown("#### 💡 Personalized Suggestions")
                        for suggestion in suggestions['suggestions']:
                            suggestion_type = get_suggestion_type(suggestion)
                            render_suggestion(suggestion, suggestion_type)
                        
                        st.markdown("#### 📈 Daily Progress After This Meal")
                        after_meal = suggestions['after_meal']
                        remaining_cal = suggestions['remaining']
                        daily_target = suggestions['daily_target']
                        
                        st.markdown(f"""
                            <div class="progress-container">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <span><strong>Calories</strong></span>
                                    <span>{after_meal['calories']:.0f} / {daily_target}</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        progress = min(after_meal['calories'] / daily_target, 1.0)
                        st.progress(progress)
                        
                        if remaining_cal['calories'] > 0:
                            st.success(f"🎉 You'll have **{remaining_cal['calories']:.0f} calories** remaining!")
                        else:
                            st.warning("⚠️ This meal would exceed your daily target")
                        
                        st.markdown("---")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("➕ Add to Today's Intake", use_container_width=True):
                                st.session_state.calories_consumed += int(nutrition['calories'])
                                st.session_state.protein_consumed += nutrition['protein']
                                st.session_state.carbs_consumed += nutrition['carbs']
                                st.session_state.fat_consumed += nutrition['fat']
                                st.rerun()
                        with col_b:
                            if st.button("🔄 Analyze Another", use_container_width=True):
                                st.rerun()
                    
                    else:
                        st.error(f"❌ {result.get('error', 'Analysis failed')}")
                
                except Exception as e:
                    st.error(f"❌ Error analyzing image: {str(e)}")
                    st.info("💡 Make sure the models are trained properly.")
        else:
            st.info("👆 Upload a food image to get started!")
            
            st.markdown("#### 🍽️ Supported Foods")
            st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
            
            food_chips_html = ""
            for food in FOOD_CATEGORIES:
                emoji = FOOD_EMOJIS.get(food, "🍽️")
                food_chips_html += f'<span class="food-category-chip">{emoji} {food.title()}</span>'
            
            st.markdown(f'<div style="text-align: center; padding: 1rem;">{food_chips_html}</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer">
            <p style="margin: 0;">🥗 <strong>AI Nutrition Advisor</strong></p>
            <p style="margin: 0.25rem 0; font-size: 0.9rem;">Powered by Machine Learning | Image Classification + Calorie Prediction</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
