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


