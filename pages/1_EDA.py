import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Exploratory Data Analysis')

# Load dataset
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

st.write('Dataset Overview:')
st.dataframe(df.head())

st.write('Statistical Summary:')
st.write(df.describe())

st.write('Missing Values:')
st.write(df.isnull().sum())

st.write('Correlation Heatmap:')
corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True)
st.pyplot(plt)
