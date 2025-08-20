import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ---------------------🧭 Streamlit Config ----------------------
# This MUST be the very first Streamlit command in your script.
st.set_page_config(page_title="Devil Cars", page_icon="🚗", layout="wide")
st.title("🚘 Devil Cars: Smart Car Price Predictor")

# ---------------------🎨 Custom CSS Styling ----------------------
st.markdown("""
    <style>
        body {
            background-color: #0f1117;
        }
        .main {
            background-color: #0f1117;
            color: white;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #d4af37;
        }
        .stButton>button {
            background-color: #d4af37;
            color: black;
            border-radius: 10px;
            font-weight: bold;
            padding: 0.6em 1.2em;
            transition: transform 0.3s ease;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background-color: #ffcc00;
        }
        .stSelectbox > div > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {
            background-color: #1e1e1e !important;
            color: #fff !important;
            transition: all 0.3s ease;
        }
        .stSelectbox:hover > div > div,
        .stNumberInput:hover > div > div > input,
        .stTextInput:hover > div > div > input {
            border: 1px solid #d4af37;
        }
        .stDataFrame {
            background-color: #1c1c1c;
        }
        section[data-testid="stSidebar"] {
            background-color: #b30000;
        }
        section[data-testid="stSidebar"]:hover {
            background-color: #990000;
        }
        section[data-testid="stSidebar"] .css-ng1t4o {
            color: white !important;
        }
        .block-container {
            background: linear-gradient(145deg, #2c2c2c, #1c1c1c);
            border-radius: 10px;
            padding: 2em;
            transition: box-shadow 0.3s ease;
        }
        .block-container:hover {
            box-shadow: 0 0 20px #d4af37;
        }
        .stForm {
            background: #2c2c2c;
            padding: 1.5em;
            border-radius: 10px;
            animation: fadeInUp 0.8s ease-out;
        }
        @keyframes fadeInUp {
            0% {
                opacity: 0;
                transform: translateY(20px);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------🚗 Load Assets ----------------------
df = pd.read_csv('Car Sell Dataset.csv')
model = joblib.load('PipelineCar.pkl')
a = df.drop(columns=['Price']).head(5)
print(model.predict(a))

# ---------------------📌 Sidebar Navigation ----------------------
page = st.sidebar.radio("📂 Menu", ["🏠 Home", "📊 Data Analysis", "🧾 Prediction"])

# ---------------------🏠 Home ----------------------
if page.startswith("🏠"):
    st.image("devil_cars_logo.png", use_container_width=True)
    st.markdown("""
        <div style="background-color:#1e1e1e;padding:1.5em;border-radius:10px">
        <h3>🔥 Welcome to <span style='color:#d4af37'>Devil Cars</span>!</h3>
        🚗 This app uses cutting-edge machine learning to predict car prices based on specifications like brand, model, year, fuel type, and more.
        </div>
    """, unsafe_allow_html=True)

# ---------------------📊 Data Analysis ----------------------
elif page.startswith("📊"):
    st.header("📈 Data Insights")
    st.markdown("""<hr style="border: 1px solid #d4af37">""", unsafe_allow_html=True)
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🧮 Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("💰 Price Distribution")
    plt.figure(figsize=(10, 5))
    sns.set_style("darkgrid")
    sns.histplot(df['Price'], bins=30, kde=True, color='#d4af37')
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Car Prices")
    st.pyplot(plt)

    st.subheader("📊 Fuel Type Proportion (Pie Chart)")
    fuel_counts = df['Fuel Type'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(fuel_counts, labels=fuel_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    ax1.set_title("Fuel Type Proportion")
    st.pyplot(fig1)

    st.subheader("📍 Average Car Price by State")
    plt.figure(figsize=(12, 6))
    state_avg = df.groupby('State')['Price'].mean().sort_values(ascending=False)
    sns.barplot(x=state_avg.index, y=state_avg.values, palette='coolwarm')
    plt.xticks(rotation=45)
    plt.xlabel("State")
    plt.ylabel("Average Price")
    plt.title("Average Car Price by State")
    st.pyplot(plt)

    st.subheader("🚘 Car Type Distribution (Histogram)")
    plt.figure(figsize=(8, 5))
    df['Car Type'].value_counts().plot(kind='bar', color=sns.color_palette('Set2'))
    plt.title("Car Type Count")
    plt.xlabel("Car Type")
    plt.ylabel("Count")
    st.pyplot(plt)

    st.subheader("🔁 Transmission Type Distribution (Pie Chart)")
    transmission_counts = df['Transmission'].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.pie(transmission_counts, labels=transmission_counts.index, autopct='%1.1f%%', colors=sns.color_palette('muted'))
    ax2.set_title("Transmission Type Share")
    st.pyplot(fig2)

# ---------------------🧾 Prediction ----------------------
elif page.startswith("🧾"):
    st.header("🧠 Car Price Prediction Engine")
    st.markdown("<p style='color:#d4af37'>Fill in the details of your car to estimate its resale value.</p>", unsafe_allow_html=True)

    with st.form(key='car_form'):
        cols1, cols2, cols3 = st.columns(3)
        with cols1:
            brand = st.selectbox("🚘 Brand", df['Brand'].unique())
            model_name = st.selectbox("📦 Model", df[df['Brand'] == brand]['Model Name'].unique())
            model_variant = st.selectbox("🔢 Model Variant", df[df['Model Name'] == model_name]['Model Variant'].unique())
            year = st.number_input("📅 Year", min_value=2000, max_value=2023, value=2020)
        with cols2:
            cartype = st.selectbox("🚗 Car Type", df['Car Type'].unique())
            fueltype = st.selectbox("⛽ Fuel Type", df['Fuel Type'].unique())
            transmission = st.selectbox("⚙️ Transmission", df['Transmission'].unique())
        with cols3:
            ownership = st.selectbox("🧍 Ownership", df['Owner'].unique())
            kilometers_driven = st.number_input("🛣️ Kilometers", min_value=0, value=10000)
            state = st.selectbox("🗺️ State", df['State'].unique())

        accidental = st.selectbox("🚨 Accidental", df['Accidental'].unique())
        submit_button = st.form_submit_button(label='🔍 Predict Price')

        if submit_button:
            input_data = pd.DataFrame([{
                'Brand': brand,
                'Model Name': model_name,
                'Year': year,
                'Model Variant': model_variant,
                'Car Type': cartype,
                'Fuel Type': fueltype,
                'Transmission': transmission,
                'Owner': ownership,
                'Kilometers': kilometers_driven,
                'State': state,
                'Accidental': accidental
            }])

            predicted_price = model.predict(input_data)[0]
            st.success(f"🎯 **Estimated Price: ₹ {predicted_price:,.2f}** 💸")