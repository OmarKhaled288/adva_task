import streamlit as st
import polars as pl
from objects import schema
from pipeline import preprocess

def run(model, le):
    st.title("Upload a CSV file")

    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            df = pl.scan_csv(uploaded_file, schema=schema, ignore_errors=True)
            st.write("First 5 rows from your file:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Could not read file: {e}\nPlease upload a valid CSV file.")

        if st.button("Predict", key="upload"):
            preprocessed_df = preprocess(df)
            prediction = model.predict(preprocessed_df)
            df = df.with_columns(pl.Series("Prediction", le.inverse_transform(prediction), pl.String)).collect()
            st.write("Predictions:")
            st.dataframe(df)
        
            csv = df.write_csv()
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
                key="download-csv"
            )
    
    else:
        st.info("Upload a file to use this page.")