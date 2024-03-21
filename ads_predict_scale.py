import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Sales Prediction App")
st.write("This app predicts the **sales** from the different channels!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.0, 300.0, 150.0)
    Radio = st.sidebar.slider('Radio', 0.0, 50.0, 10.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 115.0, 80.5)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_scaler = pickle.load(open("scaler_features-ads.pkl","rb"))
df_scaledfeatures =loaded_scaler.fit_transform(df)

loaded_model = pickle.load(open("modeladvertising.h5", "rb"))
new_pred = loaded_model.predict(df_scaledfeatures)

loaded_target= pickle.load(open("scaler_target-ads.pkl","rb"))
print(new_pred.dtype)
#df_prediction= loaded_target.inverse_transform(new_pred)

st.subheader('Predicted Sales')
#st.write(df_prediction)
