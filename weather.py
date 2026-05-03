# app.py
# Weather Data Analysis and Prediction using Streamlit
# Run:
# pip install streamlit pandas numpy matplotlib scikit-learn

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# PAGE CONFIG 
st.set_page_config(
    page_title="Weather Data Analysis and Prediction",
    page_icon="🌦️",
    layout="wide"
)

#  TITLE 
st.title("🌦️ Weather Data Analysis and Prediction")
st.write("Upload historical weather data and predict future temperature trends.")

#  SIDEBAR 
st.sidebar.header("Upload CSV File")

uploaded_file = st.sidebar.file_uploader(
    "Upload Weather CSV File",
    type=["csv"]
)

# SAMPLE DATA INFO 
st.info("""
CSV file should contain these columns:

1. Date  
2. Temperature

Example:

Date,Temperature  
2024-01-01,25  
2024-01-02,27  
2024-01-03,26
""")

#  MAIN 
if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check columns
        if "Date" not in df.columns or "Temperature" not in df.columns:
            st.error("CSV must contain Date and Temperature columns.")
            st.stop()

        # Convert date column
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        # Show dataset
        st.subheader("📄 Uploaded Dataset")
        st.dataframe(df, use_container_width=True)

        #  BASIC ANALYSIS 
        st.subheader("📊 Basic Statistics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Temp", f"{df['Temperature'].mean():.2f} °C")
        col2.metric("Maximum Temp", f"{df['Temperature'].max():.2f} °C")
        col3.metric("Minimum Temp", f"{df['Temperature'].min():.2f} °C")

        #  PLOT HISTORICAL DATA
        st.subheader("📈 Historical Temperature Trend")

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df["Date"], df["Temperature"], marker="o")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("Historical Temperature")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # PREPARE DATA FOR ML 
        df["Day_Number"] = np.arange(len(df))

        X = df[["Day_Number"]]
        y = df["Temperature"]

        model = LinearRegression()
        model.fit(X, y)

        #  FUTURE PREDICTION 
        st.subheader("🔮 Predict Future Temperature")

        future_days = st.slider(
            "Select number of future days",
            min_value=1,
            max_value=30,
            value=7
        )

        future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
        predictions = model.predict(future_X)

        future_dates = pd.date_range(
            start=df["Date"].max() + pd.Timedelta(days=1),
            periods=future_days
        )

        pred_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted Temperature": predictions
        })

        st.dataframe(pred_df, use_container_width=True)

        #  PLOT PREDICTION
        st.subheader("📉 Future Temperature Prediction")

        fig2, ax2 = plt.subplots(figsize=(10, 4))

        # Historical
        ax2.plot(
            df["Date"],
            df["Temperature"],
            label="Historical Data",
            marker="o"
        )

        # Prediction
        ax2.plot(
            pred_df["Date"],
            pred_df["Predicted Temperature"],
            label="Predicted Data",
            marker="o",
            linestyle="--"
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Temperature (°C)")
        ax2.set_title("Temperature Forecast")
        ax2.legend()

        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # DOWNLOAD
        csv = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction CSV",
            data=csv,
            file_name="weather_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.warning("Please upload a CSV file to continue.")

#  FOOTER
st.markdown("---")
st.write("Developed using Python, Streamlit, Pandas, Scikit-Learn")