import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Sales Prediction App")
st.write("This app predicts the **sales** from the different channels!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Tv = st.sidebar.slider('TV', 0.0, 300.0, 150.0)
    Radio = st.sidebar.slider('Radio', 0.0, 50.0, 10.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 115.0, 80.5)
   data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = sns.load_dataset('advertising')
X = data.drop(['Sales'],axis=1)
Y = data.Sales.copy()

modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
