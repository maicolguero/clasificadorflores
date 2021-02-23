import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Titulo

st.write("""
# Clasificando flores udenar
""") 
st.sidebar.header('ingrese los parametros para predecir')

def parametros():
    SL= st.sidebar.slider('sepalo-L',4.3,7.9,5.4)
    SW= st.sidebar.slider('sepalo-W',2.0,4.4,3.4)
    PL=st.sidebar.slider('petalo-L',1.0,6.9,1.3)
    PW= st.sidebar.slider('petalo-W',0.1,2.5,0.2)

    data= {

        'SL':SL,
        'SW':SW,
        'PL':PL,
        'PW':PW
    }

    predictores = pd.DataFrame(data,index=[0])
    return predictores
df = parametros()
iris = datasets.load_iris()
x = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(x,y)

prediccion = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.write(prediccion)

st.subheader('class labels and their corresponding index')
st.write(iris.target_names)

st.subheader('prediccion')
st.write(iris.target_names[prediccion])

st.subheader('prediction probability')
st.write(prediction_proba)