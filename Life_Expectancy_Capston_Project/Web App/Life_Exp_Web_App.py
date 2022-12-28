import streamlit as st
import pickle

log_reg=pickle.load(open("log_reg.pkl",'rb'))
rf=pickle.load(open('ran_forest.pkl','rb'))
knn=pickle.load(open('knn.pkl','rb'))
svr=pickle.load(open('svr.pkl','rb'))
dt=pickle.load(open('dt.pkl','rb'))
ada=pickle.load(open('ada_boost.pkl','rb'))
xgb=pickle.load(open('xg_boost.pkl','rb'))

st.title("Life Expectancy")
html_temp = """
    <div style="background-color:tomato ;padding:13px">
    <h2 style="color:white;text-align:center;">Life Expectancy Web App</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

activities = ['LOG_REG','RF','KNN','SVM','DT','ADABOOST','XGBOOST']

option = st.sidebar.selectbox('Please select the model of your choice',activities)
st.subheader(option)


HIV = st.slider('Select Deaths per 1000 live births HIV/AIDS (0-4 years)', min_value=0.0,max_value= 50.7)
Adult_Mortality = st.slider('Select Adult Mortality Rates of people (probability of dying between 15 and 60 years per 1000 population)', min_value=0.0, max_value=724.0)
Schooling = st.slider('Select Number of years of Schooling in years', min_value=0.0, max_value=20.8)
thinness_10_19_years = st.slider('Select Prevalence of thinness among childre for Age 10 to 19 in value equalant to percentage', min_value=0.0, max_value=27.8)
infant_deaths = st.slider('Select Number of Infant Deaths per 1000 population', min_value=0.0,max_value=1801.0)
Status = st.radio('Select Nation (Developed is 0 and developing is 1 )',[0,1])

inputs=['HIV/AIDS',
 'Adult_Mortality',
 'Schooling',
 'BMI',
 'thinness_10_19_years',
 'infant_deaths',
 'Status']


import numpy as np
ip=np.array(inputs)

if st.button('Predict'):
    if option=='LOG_REG':
        st.success(log_reg.predict(ip))
    elif option=='RF':
        st.success(rf.predict(ip))
    elif option=='KNN':
        st.success(knn.predict(ip))
    elif option=='SVM':
        st.success(svr.predict(ip))
    elif option=='DT':
        st.success(dt.predict(ip))
    elif option=='ADABOOST':
        st.success(ada.predict(ip))
    else:
        st.success(xgb.predict(ip))
