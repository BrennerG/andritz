import pandas as pd
import streamlit as st


st.title('Andritz Coding Challenge')

# INPUT 1
st.header('input 1')
inp1 = pd.read_csv("data/input1.csv", 
                   sep=";", 
                   header=0, 
                   names=['date', 'inlet_pressure', 'temperature', 'ph', 'peroxide', 'soda'])
inp1['date'] = pd.to_datetime(inp1['date'], format="%d.%m.%Y %H:%M")
st.dataframe(inp1.head(10))
inp1.dtypes
st.write(inp1.describe())

# INPUT 2
st.header('input 2')
inp2 = pd.read_csv("data/input2.csv",
                   sep=";",
                   header=0,
                   names=["date", 'inlet_brightness'])
inp2['date'] = pd.to_datetime(inp2['date'], format="%d.%m.%Y %H:%M")
st.dataframe(inp2.head(10))
inp2.dtypes
st.write(inp2.describe())

F"---\n OK so the inlet_brightness is changed every 2 hours, but the measurements are taken every 5 minutes. \n Let's check if this adds up and also if the measurements are taken regularly"
with st.echo():
    inp1.shape[0] / inp2.shape[0]

F"means 24 measurements for 2 hours - seems correct. Since we want to join datasets we will fill in the 5 minute intervals of inp1 with the according values of inp2 OR we can try to interpolate the data from inp2 according to its trend (linear, quadratic, cubic, ..., fourier)"

st.header('target')
targ = pd.read_csv("data/target.csv", 
                   sep=";",
                   header=0,
                   names=['date', 'target_brightness']
                   )
targ['date'] = pd.to_datetime(targ['date'], format="%d.%m.%Y %H:%M")
st.dataframe(targ.head(10))
targ.dtypes
st.write(targ.describe())

F"OK interesting so the targets are taken every 2 hours, naturally with a delay of 3h45min because the process takes so much time. I should visualize the time axis somehow, like what's happening when"