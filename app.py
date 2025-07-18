# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(page_title="EV Adoption Forecast", layout="centered")

st.title("🔋 EV Adoption Forecasting App")
st.write("Upload a historical EV population CSV to predict future adoption.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        st.write("📄 CSV Columns:", df.columns.tolist())

        # Step 1: Parse Date column
        df['Month'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Month'])  # Remove bad dates

        # Step 2: Clean numeric column
        df['Electric Vehicle (EV) Total'] = pd.to_numeric(
            df['Electric Vehicle (EV) Total'].astype(str).str.replace(',', '', regex=True), 
            errors='coerce'
        )
        df = df.dropna(subset=['Electric Vehicle (EV) Total'])

        # Step 3: Group and forecast
        df_grouped = df.groupby('Month')['Electric Vehicle (EV) Total'].sum().reset_index()
        df_grouped['Month_Ordinal'] = df_grouped['Month'].map(pd.Timestamp.toordinal)

        # Step 4: Linear Regression
        X = df_grouped[['Month_Ordinal']]
        y = df_grouped['Electric Vehicle (EV) Total']
        model = LinearRegression()
        model.fit(X, y)

        # Step 5: Forecast next 12 months
        future_dates = pd.date_range(start=df_grouped['Month'].max(), periods=13, freq='MS')[1:]
        future_ordinal = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1, 1)
        future_preds = model.predict(future_ordinal)
        future_df = pd.DataFrame({
            'Month': future_dates,
            'Predicted EV Population': future_preds.astype(int)
        })

        # Step 6: Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_grouped['Month'], df_grouped['Electric Vehicle (EV) Total'], label='Actual')
        ax.plot(future_df['Month'], future_df['Predicted EV Population'], label='Forecast', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("EV Count")
        ax.set_title("EV Adoption Forecast")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Step 7: Save model and output
        joblib.dump(model, "ev_forecast_model.pkl")
        future_df.to_csv("EV_Forecast_Output.csv", index=False)

        st.success("✅ Forecast complete!")
        st.download_button("📥 Download Forecast CSV", data=future_df.to_csv(index=False), file_name="EV_Forecast_Output.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📂 Please upload a CSV file to begin.")
