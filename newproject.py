import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="8th Mile to Sapthagiri Real Estate",
    page_icon="🏠",
    layout="wide"
)

# ---------------- TITLE ----------------
st.markdown("""
    <h1 style='text-align:center; color:#4B8BBE;'>
    🏠 8th Mile to Sapthagiri Real Estate Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(r"C:\Users\deeks\OneDrive\Documents\8th_mile_to_sapthagiri_real_estate.csv")

# ---------------- CLEAN DATA ----------------
df.dropna(inplace=True)

# Convert numeric columns
numeric_cols = [
    "Plot_Area_sqft",
    "Distance_to_8th_Mile_km",
    "Distance_to_Sapthagiri_College_km",
    "Road_Width_ft",
    "Price_in_Lakhs_INR"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filter Properties")

locality = st.sidebar.selectbox(
    "Select Locality",
    ["All"] + list(df["Locality"].unique())
)

facing = st.sidebar.selectbox(
    "Select Facing",
    ["All"] + list(df["Facing"].unique())
)

# Apply filters
filtered_df = df.copy()

if locality != "All":
    filtered_df = filtered_df[filtered_df["Locality"] == locality]

if facing != "All":
    filtered_df = filtered_df[filtered_df["Facing"] == facing]

# ---------------- METRICS ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "🏘 Total Properties",
    len(filtered_df)
)

col2.metric(
    "💰 Avg Price",
    f"{filtered_df['Price_in_Lakhs_INR'].mean():.2f} Lakhs"
)

col3.metric(
    "📏 Avg Area",
    f"{filtered_df['Plot_Area_sqft'].mean():.0f} sqft"
)

col4.metric(
    "🛣 Avg Road Width",
    f"{filtered_df['Road_Width_ft'].mean():.0f} ft"
)

st.markdown("---")

# ---------------- DATA TABLE ----------------
st.subheader("📋 Property Data")

st.dataframe(filtered_df, use_container_width=True)

# ---------------- CHARTS ----------------
st.subheader("📊 Visual Analysis")

chart1, chart2 = st.columns(2)

with chart1:
    fig1 = px.scatter(
        filtered_df,
        x="Plot_Area_sqft",
        y="Price_in_Lakhs_INR",
        color="Locality",
        size="Road_Width_ft",
        title="Plot Area vs Price"
    )
    st.plotly_chart(fig1, use_container_width=True)

with chart2:
    fig2 = px.bar(
        filtered_df.groupby("Locality")["Price_in_Lakhs_INR"].mean().reset_index(),
        x="Locality",
        y="Price_in_Lakhs_INR",
        color="Locality",
        title="Average Price by Locality"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- DISTANCE ANALYSIS ----------------
st.subheader("📍 Distance Analysis")

chart3, chart4 = st.columns(2)

with chart3:
    fig3 = px.histogram(
        filtered_df,
        x="Distance_to_8th_Mile_km",
        nbins=15,
        title="Distance to 8th Mile"
    )
    st.plotly_chart(fig3, use_container_width=True)

with chart4:
    fig4 = px.histogram(
        filtered_df,
        x="Distance_to_Sapthagiri_College_km",
        nbins=15,
        title="Distance to Sapthagiri College"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------- PRICE PREDICTION ----------------
st.markdown("---")
st.subheader("🤖 Property Price Prediction")

# Features
X = df[[
    "Plot_Area_sqft",
    "Distance_to_8th_Mile_km",
    "Distance_to_Sapthagiri_College_km",
    "Road_Width_ft"
]]

y = df["Price_in_Lakhs_INR"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)

st.success(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# ---------------- USER INPUT ----------------
st.write("### Enter Property Details")

input1, input2 = st.columns(2)

with input1:
    area = st.number_input(
        "Plot Area (sqft)",
        min_value=300,
        max_value=10000,
        value=1200
    )

    dist_8mile = st.number_input(
        "Distance to 8th Mile (km)",
        min_value=0.0,
        max_value=20.0,
        value=2.0
    )

with input2:
    dist_saptha = st.number_input(
        "Distance to Sapthagiri College (km)",
        min_value=0.0,
        max_value=20.0,
        value=2.0
    )

    road = st.number_input(
        "Road Width (ft)",
        min_value=10.0,
        max_value=100.0,
        value=30.0
    )

# Prediction button
if st.button("🔮 Predict Price"):
    
    prediction = model.predict([[
        area,
        dist_8mile,
        dist_saptha,
        road
    ]])

    st.balloons()

    st.markdown(f"""
        <div style="
            background-color:#dff0d8;
            padding:25px;
            border-radius:15px;
            text-align:center;
        ">
            <h2>🏡 Predicted Property Price</h2>
            <h1 style='color:green;'>
                ₹ {prediction[0]:.2f} Lakhs
            </h1>
        </div>
    """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<center>
<h4>✨ Smart Real Estate Analytics System</h4>
<p>Developed using Python, Streamlit & Machine Learning</p>
</center>
""", unsafe_allow_html=True)