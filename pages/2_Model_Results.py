import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Model Evaluation')

# Load model
model = joblib.load('model/stroke_model.pkl')

# Load dataset
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')
df.dropna(inplace=True)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Predictions
y_pred = model.predict(X)

# Classification report
st.write('Classification Report:')
st.text(classification_report(y, y_pred))

# Confusion matrix
st.write('Confusion Matrix:')
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
st.pyplot(plt)
