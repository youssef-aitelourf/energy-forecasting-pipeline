"""
Energy Consumption Forecasting - Streamlit Application
Author: Youssef AIT ELOURF
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Energy Forecasting",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model_path = Path("models/best_model.pkl")
        scaler_path = Path("models/scaler.pkl")
        feature_names_path = Path("models/feature_names.pkl")
        
        if not model_path.exists():
            st.error(f"Model not found at {model_path}")
            return None, None, None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        
        feature_names = None
        if feature_names_path.exists():
            with open(feature_names_path, 'rb') as f:
                feature_names = pickle.load(f)
        
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def create_feature_inputs():
    """Create input fields for all features"""
    st.sidebar.header("🎛️ Input Parameters")
    
    features = {}
    
    with st.sidebar.expander("⚡ Power Consumption", expanded=True):
        features['Appliances'] = st.number_input("Appliances (Wh)", min_value=0.0, max_value=2000.0, value=60.0, step=1.0)
        features['lights'] = st.number_input("Lights (Wh)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    
    with st.sidebar.expander("🌡️ Temperature Sensors", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            features['T1'] = st.number_input("Kitchen (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)
            features['T2'] = st.number_input("Living Room (°C)", min_value=-10.0, max_value=50.0, value=21.0, step=0.1)
            features['T3'] = st.number_input("Laundry (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
            features['T4'] = st.number_input("Office (°C)", min_value=-10.0, max_value=50.0, value=21.0, step=0.1)
        with col2:
            features['T5'] = st.number_input("Bathroom (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
            features['T6'] = st.number_input("North Side (°C)", min_value=-10.0, max_value=50.0, value=19.0, step=0.1)
            features['T7'] = st.number_input("Ironing Room (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
            features['T8'] = st.number_input("Teenager Room (°C)", min_value=-10.0, max_value=50.0, value=21.0, step=0.1)
    
    with st.sidebar.expander("💧 Humidity Sensors", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            features['RH_1'] = st.number_input("Kitchen Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            features['RH_2'] = st.number_input("Living Room Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            features['RH_3'] = st.number_input("Laundry Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            features['RH_4'] = st.number_input("Office Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        with col2:
            features['RH_5'] = st.number_input("Bathroom Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
            features['RH_6'] = st.number_input("North Side Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
            features['RH_7'] = st.number_input("Ironing Room Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
            features['RH_8'] = st.number_input("Teenager Room Humidity (%)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    
    with st.sidebar.expander("🌍 External Conditions", expanded=True):
        features['T_out'] = st.number_input("Outside Temperature (°C)", min_value=-30.0, max_value=50.0, value=15.0, step=0.1)
        features['Press_mm_hg'] = st.number_input("Pressure (mm Hg)", min_value=700.0, max_value=800.0, value=760.0, step=1.0)
        features['RH_out'] = st.number_input("Outside Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        features['Windspeed'] = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        features['Visibility'] = st.number_input("Visibility (km)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
        features['Tdewpoint'] = st.number_input("Dew Point (°C)", min_value=-30.0, max_value=30.0, value=10.0, step=0.1)
    
    with st.sidebar.expander("📅 Time Information", expanded=True):
        features['WeekStatus'] = st.selectbox("Week Status", options=[0, 1], format_func=lambda x: "Weekend" if x == 1 else "Weekday")
        features['Day_of_week'] = st.selectbox("Day of Week", options=list(range(7)), 
                                               format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])
    
    return features

def make_prediction(model, scaler, feature_names, features):
    """Make prediction using the loaded model"""
    try:
        # Create DataFrame with features
        df = pd.DataFrame([features])
        
        # Scale features if scaler is available (before adding missing features)
        if scaler is not None:
            # Get scaler's expected features
            scaler_features = scaler.feature_names_in_
            
            # Add missing scaler features with 0
            for col in scaler_features:
                if col not in df.columns:
                    df[col] = 0
            
            # Scale only the features the scaler knows about
            df_to_scale = df[scaler_features].copy()
            df_scaled = pd.DataFrame(
                scaler.transform(df_to_scale),
                columns=scaler_features,
                index=df.index
            )
            
            # Replace scaled values in original dataframe
            for col in scaler_features:
                df[col] = df_scaled[col]
        
        # Now ensure all model features are present
        if feature_names is not None:
            # Add missing features with 0
            for col in feature_names:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns to match training
            df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return prediction, df
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

def plot_prediction_gauge(prediction):
    """Create a gauge chart for the prediction"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Energy Consumption (Wh)", 'font': {'size': 24}},
        delta={'reference': 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#1f77b4"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#90EE90'},
                {'range': [50, 100], 'color': '#FFD700'},
                {'range': [100, 200], 'color': '#FF6B6B'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150}}))
    
    fig.update_layout(height=400, font={'family': "Arial"})
    return fig

def plot_feature_importance(df):
    """Create a bar chart of feature values"""
    # Get top 10 features by absolute value
    feature_values = df.iloc[0].abs().sort_values(ascending=True).tail(10)
    
    fig = px.bar(
        x=feature_values.values,
        y=feature_values.index,
        orientation='h',
        title="Top 10 Feature Values",
        labels={'x': 'Value', 'y': 'Feature'},
        color=feature_values.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def main():
    # Header
    st.markdown('<p class="main-header">🔋 Energy Consumption Forecasting</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced ML-powered predictions for smart energy management</p>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, feature_names = load_model()
    
    if model is None:
        st.error("⚠️ Model could not be loaded. Please check if the model file exists.")
        st.info("💡 Run `python scripts/train_pipeline.py` to train and save the model first.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        ### 📊 About This App
        
        This application uses a **XGBoost machine learning model** to predict energy consumption based on:
        
        - 🏠 Indoor temperature & humidity
        - 🌍 Weather conditions
        - ⚡ Appliance usage
        - 📅 Time patterns
        
        **Adjust the parameters in the sidebar** and click **Predict** to see the forecast!
        """)
        
        # Model info
        st.success(f"""
        ### 🤖 Model Info
        - **Algorithm**: {type(model).__name__}
        - **Features**: {len(feature_names) if feature_names else 'Unknown'}
        - **Status**: ✅ Loaded
        """)
    
    with col2:
        st.markdown("### 🎯 Prediction Results")
        
        # Get feature inputs
        features = create_feature_inputs()
        
        # Predict button
        if st.button("🚀 Predict Energy Consumption", type="primary"):
            with st.spinner("Making prediction..."):
                prediction, df = make_prediction(model, scaler, feature_names, features)
                
                if prediction is not None:
                    # Display prediction
                    st.success(f"### Predicted Energy Consumption: **{prediction:.2f} Wh**")
                    
                    # Create two columns for visualizations
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        # Gauge chart
                        fig_gauge = plot_prediction_gauge(prediction)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with viz_col2:
                        # Feature importance
                        fig_features = plot_feature_importance(df)
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Interpretation
                    st.markdown("---")
                    st.markdown("### 💡 Interpretation")
                    
                    if prediction < 50:
                        st.success("✅ **Low consumption** - Very efficient energy usage!")
                    elif prediction < 100:
                        st.info("ℹ️ **Moderate consumption** - Normal energy usage pattern.")
                    else:
                        st.warning("⚠️ **High consumption** - Consider energy-saving measures.")
                    
                    # Export feature
                    st.markdown("---")
                    st.markdown("### 📥 Export Prediction")
                    result_df = pd.DataFrame([{
                        'Prediction (Wh)': prediction,
                        **features
                    }])
                    
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Prediction as CSV",
                        data=csv,
                        file_name=f"energy_prediction_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>Energy Forecasting Pipeline</strong></p>
        <p>Developed by <strong>Youssef AIT ELOURF</strong> | 
        <a href="https://github.com/youssef-aitelourf/energy-forecasting-pipeline" target="_blank">GitHub</a> | 
        <a href="mailto:youssefaitelourf@gmail.com">Contact</a></p>
        <p style="font-size: 0.8rem;">Powered by XGBoost, Streamlit, and ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
