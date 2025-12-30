import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(page_title="Gold Data Analysis App", layout="centered")

# =====================
# Load Data
# =====================
FILE_PATH = "gold_data_cleaned_pca.csv"  # Make sure the CSV is in the same folder
df = pd.read_csv(FILE_PATH)
numeric_cols = df.select_dtypes(include="number").columns

# =====================
# User Interface
# =====================
st.title("🟡 Gold Data Analysis App")
st.write("أدخل قيم المتغيرات لتحليل بيانات الذهب")

col1, col2 = st.columns(2)

with col1:
    feature_1 = st.selectbox("📊 Feature 1", numeric_cols)
    value_1 = st.number_input("Value for Feature 1", value=0.0)

with col2:
    feature_2 = st.selectbox("📊 Feature 2", numeric_cols)
    value_2 = st.number_input("Value for Feature 2", value=0.0)

# =====================
# PCA Analysis
# =====================
if st.button("📈 Run PCA Analysis"):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numeric_cols])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    st.subheader("PCA Scatter Plot")
    st.scatter_chart(pca_df)

# =====================
# Simple Prediction (Demo)
# =====================
if st.button("🔮 Predict (Demo)"):
    result = round(value_1 * 0.5 + value_2 * 0.3, 2)
    st.success(f"📌 Result based on selected features: {result}")