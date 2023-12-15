import pandas as pd
import streamlit as st


st.title('Data Preparation')

# READING DATA
with st.echo():
    inp1 = pd.read_csv("data/input1.csv", sep=";", header=0, names=['date', 'inlet_pressure', 'temperature', 'ph', 'peroxide', 'soda'])
    inp1['date'] = pd.to_datetime(inp1['date'], format="%d.%m.%Y %H:%M")
    inp2 = pd.read_csv("data/input2.csv", sep=";", header=0, names=["date", 'inlet_brightness'])
    inp2['date'] = pd.to_datetime(inp2['date'], format="%d.%m.%Y %H:%M")
    targ = pd.read_csv("data/target.csv", sep=";", header=0, names=['date', 'target_brightness'])
    targ['date'] = pd.to_datetime(targ['date'], format="%d.%m.%Y %H:%M")

# MERGING DATAFRAMES
st.markdown('''
### Merging Dataframes + Imputation
We want to join dataframes so we can use inlet_brightness like the other features.  
We can simply impute the mean values for ``inlet_brightness`` since we haven't found any trends in the data exploration. 
''')
with st.echo():
    merged = pd.merge(inp1, inp2, on='date', how='outer')
    merged['inlet_brightness'] = merged['inlet_brightness'].interpolate(method='linear')

st.dataframe(merged)
# st.write(merged.describe())

st.markdown('''
### Downsampling
Unfortunately the measurements for our target variable ``target_brightness`` are the finest granulated value we can go down to without imputing the target variable, which we will totally do if we have the time. However as a baseline we will downsample by only allowing values of the merged dataset where we have measurements of our target variable.
''')

with st.echo():
    downsampled = pd.merge(merged, targ, on='date', how='inner')
    st.dataframe(downsampled)
    st.write(downsampled.describe())

st.markdown('''
### Lags / Steps
- __Lead Time__ : 2h = 1 lag
- __Forecast Horizon__ : 2h = 1 step

this means we will look 1 step backwards and forwards for a prediction
''')

with st.echo():
    num_lags = 1
    for i in range(1, num_lags + 1):
        downsampled[f'y_lag_{i}'] = downsampled['target_brightness'].shift(i)
    
    num_steps = 1
    for i in range(1, num_steps+ 1):
        downsampled[f'y_step_{i}'] = downsampled['target_brightness'].shift(-i)

    st.dataframe(downsampled)