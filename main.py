import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Streamlit Dashboard", layout="wide")

def preprocess_data(df):
    st.subheader("Preprocessing Options")
    # 결측치 처리
    if st.checkbox("Fill missing values with column mean (numeric only)"):
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    if st.checkbox("Drop rows with any missing values"):
        df = df.dropna()
    # 범주형 변수 인코딩
    if st.checkbox("Encode categorical columns (one-hot)"):
        cat_cols = df.select_dtypes(include='object').columns
        df = pd.get_dummies(df, columns=cat_cols)
    return df

def visualize_data(df):
    st.subheader("Visualization")
    if st.checkbox("Show Histogram"):
        col = st.selectbox("Select column for histogram", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    if st.checkbox("Show Pairplot (sample 200 rows)"):
        sample_df = df.sample(min(200, len(df)), random_state=42)
        fig = sns.pairplot(sample_df)
        st.pyplot(fig)

def main():
    st.title("Streamlit Dashboard App (All-in-One)")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.write("### Data Info")
        st.write(df.describe())
        st.write(df.info())
        # 전처리
        df_processed = preprocess_data(df)
        st.write("### Processed Data Preview")
        st.dataframe(df_processed.head())
        # 시각화
        visualize_data(df_processed)
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()