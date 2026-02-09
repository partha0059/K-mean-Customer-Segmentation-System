import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Config
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css("style.css")
except FileNotFoundError:
    st.warning("style.css not found. Glassmorphic styles might be missing.")

# Load Model and Scaler
@st.cache_resource
def load_model():
    model = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.joblib.pkl')
    return model, scaler

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Model or Scaler not found. Please run 'setup_model.py' first.")
    st.stop()

# Function to map model clusters to User-Friendly IDs (0-4)
def get_cluster_mapping(kmeans_model, scaler):
    """
    Analyzes centroids to create a mapping from Model Cluster ID -> User Friendly ID (0-4).
    Target Mapping based on User Request:
    0: Low Income, Low Spending (Sensible)
    1: Low Income, High Spending (Careless)
    2: Medium Income, Medium Spending (Standard)
    3: High Income, Low Spending (Frugal)
    4: High Income, High Spending (Elite)
    """
    centers_scaled = kmeans_model.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    
    # Create a DataFrame for easy logic
    df_centers = pd.DataFrame(centers, columns=['Income', 'Score'])
    df_centers['model_id'] = range(len(df_centers))
    
    mapping = {}
    
    # 1. Standard (Medium Income, Medium Score) - usually the middle cluster
    # Closest to mean (approx 60, 50)
    # We can find the one with Income closest to mean income of centroids
    # But explicitly: 
    # Standard: Income ~40-70, Score ~40-60
    
    # Let's sort by Income first
    df_sorted = df_centers.sort_values('Income').reset_index(drop=True)
    
    # The two lowest incomes are "Low Income" group
    low_income = df_sorted.iloc[:2] 
    # The lowest Score in this group is Group 0 (Low, Low)
    id_0 = low_income.sort_values('Score').iloc[0]['model_id']
    # The highest Score in this group is Group 1 (Low, High) -> "Careless"
    id_1 = low_income.sort_values('Score').iloc[1]['model_id']
    
    # The two highest incomes are "High Income" group
    high_income = df_sorted.iloc[-2:]
    # The lowest Score in this group is Group 3 (High, Low) -> "Frugal"
    id_3 = high_income.sort_values('Score').iloc[0]['model_id']
    # The highest Score in this group is Group 4 (High, High) -> "Elite"
    id_4 = high_income.sort_values('Score').iloc[1]['model_id']
    
    # The remaining one is Standard
    remaining_ids = set(df_centers['model_id']) - {id_0, id_1, id_3, id_4}
    id_2 = list(remaining_ids)[0]
    
    mapping = {
        int(id_0): {"label": "Sensible Customer", "id": 0, "desc": "Low Income, Low Spending. Focus on value and discounts."},
        int(id_1): {"label": "Careless Spender", "id": 1, "desc": "Low Income, High Spending. Target w/ impulse deals (Caution: Credit Risk)."},
        int(id_2): {"label": "Standard Customer", "id": 2, "desc": "Average Income, Average Spending. Maintain engagement."},
        int(id_3): {"label": "Frugal / Target", "id": 3, "desc": "High Income, Low Spending. Upsell value proposition."},
        int(id_4): {"label": "Elite / Target", "id": 4, "desc": "High Income, High Spending. Premium offers and loyalty programs."}
    }
    
    return mapping

cluster_mapping = get_cluster_mapping(model, scaler)

# Sidebar
st.sidebar.markdown("### 🛠️ User Input")
st.sidebar.markdown("Enter customer details below:")

annual_income = st.sidebar.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50, step=1)
spending_score = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

# Developer Info in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="developer-card">
    <h4>👨‍💻 Developer</h4>
    <p><strong>Partha Sarathi R</strong></p>
    <div style="margin-top: 10px;">
        <span class="tech-badge">Python</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">ML</span>
        <span class="tech-badge">K-Means</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Page
st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.title("🛍️ Customer Segmentation System")
st.markdown("""
This application uses **K-Means Clustering** to group customers based on their **Annual Income** and **Spending Score**.
Use the sidebar to input customer data and detect their segment (Groups 0-4).
""")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction Logic
if st.button("Predict Segment"):
    # Prepare input
    input_data = np.array([[annual_income, spending_score]])
    
    # Scale input
    scaled_data = scaler.transform(input_data)
    
    # Predict
    model_cluster_id = model.predict(scaled_data)[0]
    
    # Get mapped info
    info = cluster_mapping.get(int(model_cluster_id))
    user_id = info['id']
    label = info['label']
    advice = info['desc']
    
    # Color logic
    color = "#00d2ff"
    if user_id == 4: color = "#ffd700" # Gold for Elite
    if user_id == 1: color = "#ff6b6b" # Redish for Careless
    
    # Display Results
    st.markdown(f"""
    <div class="glass-container" style="border-left: 5px solid {color};">
        <h2 style="margin-bottom: 0;">Group {user_id}: <span style="color: {color};">{label}</span></h2>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <h3>💡 Customer Profile</h3>
        <p style="font-size: 1.1em;">{advice}</p>
        <div style="margin-top: 20px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
            <strong>Input Data:</strong> Income: ${annual_income}k | Spending Score: {spending_score}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Explanation Section
    with st.expander("ℹ️ Why this group? (See Centroid Distances)"):
        st.write("The model assigns the group based on which cluster center (Centroid) is closest to your input.")
        
        # Calculate distances to all centroids
        distances = model.transform(scaled_data)[0]
        
        # Create a nice dataframe for display
        centers_original = scaler.inverse_transform(model.cluster_centers_)
        explanation_data = []
        
        for mid, dist in enumerate(distances):
            # Get the mapped user ID for this model ID
            map_info = cluster_mapping.get(mid)
            uid = map_info['id']
            ulabel = map_info['label']
            
            centroid_income = centers_original[mid][0]
            centroid_score = centers_original[mid][1]
            
            is_selected = (mid == model_cluster_id)
            status = "✅ Selected (Closest)" if is_selected else ""
            
            explanation_data.append({
                "User Group": f"Group {uid}: {ulabel}",
                "Centroid Income ($k)": f"{centroid_income:.1f}",
                "Centroid Score": f"{centroid_score:.1f}",
                "Distance": f"{dist:.4f}",
                "Status": status
            })
            
        df_explain = pd.DataFrame(explanation_data).sort_values("Distance")
        st.dataframe(df_explain, use_container_width=True, hide_index=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; opacity: 0.6;">
    <p>© 2026 Customer Segmentation Project | Partha Sarathi R</p>
</div>
""", unsafe_allow_html=True)
