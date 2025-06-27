import streamlit as st
import joblib
import pandas as pd

# Load trained model and training columns
model = joblib.load('laptop_price_model.pkl')
model_columns = joblib.load('model_columns.pkl')  # this must be saved during training

st.title("💻 Laptop Price Predictor")

# Input fields
company = st.selectbox("Company", ["Apple", "Dell", "HP", "Asus", "Acer", "MSI", "Lenovo", "Toshiba"])
typename = st.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation"])
ram = st.slider("RAM (GB)", 2, 64, step=2)
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, step=0.1)
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
ips = st.selectbox("IPS Display", ["No", "Yes"])
ppi = st.number_input("PPI (Pixel Density)", min_value=90.0, max_value=400.0)
cpu_brand = st.selectbox("CPU Brand", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Processor"])
hdd = st.slider("HDD (GB)", 0, 2000, step=128)
ssd = st.slider("SSD (GB)", 0, 2000, step=128)
gpu_brand = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
os = st.selectbox("Operating System", ["Windows", "Mac", "Linux", "No OS"])

# Create input DataFrame
input_df = pd.DataFrame([{
    "Company": company,
    "TypeName": typename,
    "Ram": ram,
    "Weight": weight,
    "Touchscreen": 1 if touchscreen == "Yes" else 0,
    "Ips": 1 if ips == "Yes" else 0,
    "Ppi": ppi,
    "Cpu": cpu_brand,
    "HDD": hdd,
    "SSD": ssd,
    "Gpu": gpu_brand,
    "OpSys": os
}])

# One-hot encode and align columns
input_encoded = pd.get_dummies(input_df)
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"💰 Predicted Laptop Price: ₹{int(prediction):,}")

