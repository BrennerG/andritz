import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression



st.title('Andritz Coding Challenge')
pd.set_option('display.date_dayfirst', True)  # Customize based on your desired format


# INPUT 1
st.header('input 1')
inp1 = pd.read_csv("data/input1.csv", 
                   sep=";", 
                   header=0, 
                   names=['date', 'inlet_pressure', 'temperature', 'ph', 'peroxide', 'soda'])
inp1['date'] = pd.to_datetime(inp1['date'], format="%d.%m.%Y %H:%M")
st.dataframe(inp1.head(10))
#inp1.dtypes
#st.write(inp1.describe())


# INPUT 2
st.header('input 2')
inp2 = pd.read_csv("data/input2.csv",
                   sep=";",
                   header=0,
                   names=["date", 'inlet_brightness'])
inp2['date'] = pd.to_datetime(inp2['date'], format="%d.%m.%Y %H:%M")
st.dataframe(inp2.head(10))
#st.write(inp2.describe())

st.markdown('''
OK so the inlet_brightness is either measured or changed every 2 hours, but the measurements are taken every 5 minutes. First of all why? There probably is a reason for this.  
Let's check if this adds up and also if the measurements are taken regularly.
''')

with st.echo():
    inp1.shape[0] / inp2.shape[0]

st.markdown('''
means 24 measurements for 2 hours - seems correct. Since we want to join datasets we will fill in the 5 minute intervals of inp1 with the according values of inp2 OR we can try to interpolate the data from inp2 according to its trend (linear, quadratic, cubic, ..., fourier)
''')

# INLET BRIGHTNESS RAW PLOT
st.markdown('### inlet brightness raw')
with st.echo():
    st.line_chart(data=inp2, x='date', y='inlet_brightness')

F"No appearant trend observable. let's see if there's a trend using moving averages"

# INLET BRIGHTNESS MOVING AVERAGE
st.markdown('### inlet brightness moving average')
inp2['step'] = np.arange(len(inp2.index))
with st.echo():
    moving_average = inp2['inlet_brightness'].rolling(
        window=100,          # window
        center=True,        # puts the average at the center of the window
        min_periods=50,     # choose about half the window size
    ).mean()                # compute the mean (could also do median, std, min, max, ...)
st.line_chart(data=moving_average)

# FITTING INLET BRIGHTNESS
st.markdown('### fit inlet brightness via time using linear regression')
with st.echo():
    X = inp2.loc[:, ['step']]
    y = inp2.loc[:, 'inlet_brightness']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index, name='lin_pred')

linreg_viz_df = pd.merge(inp2, y_pred.to_frame(), left_index=True, right_index=True)
st.line_chart(data=linreg_viz_df, x='date', y=['inlet_brightness', 'lin_pred'])

st.markdown('''
There is no trend whatsoever we can easily and savely do mean interpolation or sth simple for the merge.
Still I would ask stakeholders/experts here wether the imputation makes sense at all and why the measurements are only taken at these intervals.
This must be a conscious design choice probably due to the inner workings of the machines...  

We also need to impute this because the points of measurement of our target variable don't align with those of the inlet_brightness variable... WHY?! :fist:
''')

# TARGET
st.header('target')
targ = pd.read_csv("data/target.csv", 
                   sep=";",
                   header=0,
                   names=['date', 'target_brightness']
                   )
targ['date'] = pd.to_datetime(targ['date'], format="%d.%m.%Y %H:%M")

st.dataframe(targ.head(10))
st.write(targ.describe())
st.markdown('''
OK interesting so the targets are taken every 2 hours, naturally with a delay of 3h45min because the process takes so much time.
### Thinking about Origin, Lead Time and Forecast Horizon
- __Origin__: 1.Sep.2020 is the first measurement
- __Lead Time__: Since the first measurement is 3h45min after the Origin, our lead time is 3h45min or __33 steps__ (since we measure every 5 minutes). This means every prediction at a given point tries to predict a point 3h45 in the future.
- __Forecast Horizon__: Now here's the problem: Our measurements are taken _every 2hours only_. This means our ground truth's granularity is not fine enough.
Ultimately we could thus try to:
    1. downsample to the granularity of the target variable
    2. aggregate our features over the target window (e.g. mean)
    3. impute the target variable to make finer granulated predictions (in this case I would need to ask stakeholders whether this is something they want (e.g. 'live monitoring' of sorts) or if it's ok to have predictions only for every 2 hours!)
''')

st.markdown("### target brightness raw")
st.line_chart(data=targ, x='date', y='target_brightness')

st.markdown("### target brightness moving avg")
targ_viz = targ.copy()
targ_viz['step'] = np.arange(len(targ_viz.index))
with st.echo():
    moving_average = targ_viz['target_brightness'].rolling(
        window=100,          # window
        center=True,        # puts the average at the center of the window
        min_periods=50,     # choose about half the window size
    ).mean()                # compute the mean (could also do median, std, min, max, ...)
st.line_chart(data=moving_average)

st.markdown("### fit target via step using linear regression")
with st.echo():
    X = targ_viz.loc[:, ['step']]
    y = targ_viz.loc[:, 'target_brightness']
    model = LinearRegression()
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index, name='lin_pred')

targ_viz = pd.merge(targ_viz, y_pred.to_frame(), left_index=True, right_index=True)
st.line_chart(data=targ_viz, x='date', y=['target_brightness', 'lin_pred'])

st.markdown('''### Seasonality \ndoes not make any sense right? omit for now...''')

st.markdown('### Serial Dependence')
with st.echo():
    targ_viz['lag_1'] = targ_viz['target_brightness'].shift(1)
    X = targ_viz.loc[:, ['lag_1']]
    X.dropna(inplace=True)  # drop missing values in the feature set
    y = targ_viz.loc[:, 'target_brightness']  # create the target
    y, X = y.align(X, join='inner')  # drop corresponding values in target

    model = LinearRegression()
    model.fit(X, y)

    y_pred = pd.Series(model.predict(X), index=X.index, name='lag_pred')

    targ_viz = pd.merge(targ_viz, y_pred.to_frame(), left_index=True, right_index=True)
    fig = px.scatter(targ_viz, x='lag_1', y='target_brightness', title='Auto Correlation lag=1')
    fig.add_trace(go.Scatter(x=targ_viz['lag_1'], y=targ_viz['lag_pred'], mode='lines', name='linear fit'))
    st.plotly_chart(fig)

st.markdown('### Serial Dependence: Autocorrelation Plot')
with st.echo():
    lags = range(1, 33)  # Choose the number of lags to include in the plot
    autocorrelation_values = [targ_viz['target_brightness'].autocorr(lag=lag) for lag in lags]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(lags), y=autocorrelation_values, mode='markers+lines', name='Autocorrelation'))
    fig.update_layout(title='Autocorrelation Plot', xaxis_title='Lag', yaxis_title='Autocorrelation')
    st.plotly_chart(fig)