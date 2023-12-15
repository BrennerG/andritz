import pandas as pd
import streamlit as st


# READING DATA
inp1 = pd.read_csv("data/input1.csv", sep=";", header=0, names=['date', 'inlet_pressure', 'temperature', 'ph', 'peroxide', 'soda'])
inp1['date'] = pd.to_datetime(inp1['date'], format="%d.%m.%Y %H:%M")
inp2 = pd.read_csv("data/input2.csv", sep=";", header=0, names=["date", 'inlet_brightness'])
inp2['date'] = pd.to_datetime(inp2['date'], format="%d.%m.%Y %H:%M")

# MERGING DATAFRAMES
with st.echo():
    inp2.drop('step', axis=1, inplace=True)
    merged = pd.merge(inp1, inp2, on='date', how='outer')
    merged['inlet_brightness'] = merged['inlet_brightness'].interpolate(method='linear')

st.dataframe(merged)
st.write(merged.describe())