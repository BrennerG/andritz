import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score



st.title('Coding Challenge : Gabriel Breiner')
st.markdown('''
https://github.com/BrennerG/andritz

As part of our selection process, we have designed a challenge for the second round of assessments.
Enclosed is a PDF attachment outlining a flowchart of a bleaching process, complete with various input and output variables as csv files.
Your task is to develop an effective machine learning model to accurately __predict the specified target variable__.

Your submission should address the following key areas:

__1 Feature Analysis__: Identify and justify which input features are most crucial in predicting the target variable.  
__2 Model Development__: Train a machine learning model capable of predicting the target variable from the identified input features.  
__3 Result Presentation__: Graphically present your findings using appropriate methods such as plots, heat maps, etc.  
__4 Code Submission__: Include the complete code of your work, ensuring it is executable for our review.  
 
We eagerly anticipate your participation in this challenge and look forward to receiving your submission by __Tuesday, 19th December, before 1 PM__. Following your submission, we will arrange a Microsoft Teams meeting to discuss your findings in detail.
''')
st.divider()
# TODO shorten the markdown comments / description to bullet points!


st.markdown('''
# 1 Data Exploration
''')
with st.echo():
    inp1_features = ['date', 'inlet_pressure', 'temperature', 'ph', 'peroxide', 'soda']
    inp2_features = ['date', 'inlet_brightness']
    targ_features = ['date', 'target_brightness']

    # reading csv files
    inp1 = pd.read_csv("data/input1.csv", sep=";", header=0, names=inp1_features)
    inp1['date'] = pd.to_datetime(inp1['date'], format="%d.%m.%Y %H:%M")
    inp2 = pd.read_csv("data/input2.csv", sep=";", header=0, names=inp2_features)
    inp2['date'] = pd.to_datetime(inp2['date'], format="%d.%m.%Y %H:%M")
    targ = pd.read_csv("data/target.csv", sep=";", header=0, names=targ_features)
    targ['date'] = pd.to_datetime(targ['date'], format="%d.%m.%Y %H:%M")

st.markdown('#### Input 1')
st.dataframe(inp1.head(10))
st.markdown('#### Input 2')
st.dataframe(inp2.head(10))
st.markdown('#### Target')
st.dataframe(targ.head(10))

st.markdown('## 1.1 Stationarity')
st.markdown('### Feature Set 1')
selected_feature = st.selectbox(
    "Moving Average for:",
    [x for x in inp1_features if x != 'date'],
    index=0,
)
window = st.slider("moving average (window)", 1, 3000, 288)

with st.echo(): # TODO write a function for this - repeating this code is embarassing!
    ma_viz = inp1.copy()
    ma_viz['step'] = np.arange(len(ma_viz.index))

    # create moving average
    moving_average = ma_viz[selected_feature].rolling(
        window=window,      
        center=True,        
        min_periods=round(window/2),     
    ).mean()
    ma_viz['moving_avg_'+selected_feature] = moving_average

    # linear regression
    X = ma_viz.loc[:, ['step']]
    y = ma_viz.loc[:, 'moving_avg_'+selected_feature]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = pd.Series(model.predict(X), index=X.index, name='lin_pred')
    linreg_viz_df = pd.merge(ma_viz, y_pred.to_frame(), left_index=True, right_index=True)

st.plotly_chart(px.line(linreg_viz_df, x='date', y=['moving_avg_'+selected_feature, 'lin_pred']))
st.metric(F'coefficient: "timestep" for "{selected_feature}"', model.coef_[0])
#st.plotly_chart(px.line(ma_viz, x='date', y='moving_avg_'+selected_feature))

st.divider()
st.markdown('### Feature Set 2 (inlet_brightness)')
window2 = st.slider("feature set 2: moving average (window)", 1, 100, 6)
ma_viz2 = inp2.copy()
ma_viz2['step'] = np.arange(len(ma_viz2.index))
moving_average = ma_viz2['inlet_brightness'].rolling(
    window=window2,      
    center=True,        
    min_periods=round(window2/2),     
).mean()
ma_viz2['inlet_brightness'] = moving_average

# linear regression: step -> inlet_brightness
X = ma_viz2.loc[:, ['step']]
y = ma_viz2.loc[:, 'inlet_brightness']
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index, name='lin_pred')
linreg_viz_df = pd.merge(ma_viz2, y_pred.to_frame(), left_index=True, right_index=True)

#st.line_chart(data=linreg_viz_df, x='date', y=['inlet_brightness', 'lin_pred'])
st.plotly_chart(px.line(linreg_viz_df, x='date', y=['inlet_brightness', 'lin_pred']))
st.metric(F'coefficient: "timestep" for "inlet_brightness"', model.coef_[0])


st.divider()
st.markdown('### Target Variable (target_brightness)')
window3 = st.slider("target: moving average (window)", 1, 100, 6)
targ_viz = targ.copy()
targ_viz['step'] = np.arange(len(targ_viz.index))
moving_average = targ_viz['target_brightness'].rolling(
    window=window3,      
    center=True,        
    min_periods=round(window3/2),     
).mean()
targ_viz['target_brightness'] = moving_average

# linear regression: step -> inlet_brightness
X = targ_viz.loc[:, ['step']]
y = targ_viz.loc[:, 'target_brightness']
model = LinearRegression()
model.fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index, name='lin_pred')
linreg_viz_df = pd.merge(targ_viz, y_pred.to_frame(), left_index=True, right_index=True)

#st.line_chart(data=linreg_viz_df, x='date', y=['inlet_brightness', 'lin_pred'])
st.plotly_chart(px.line(linreg_viz_df, x='date', y=['target_brightness', 'lin_pred']))
st.metric(F'coefficient: "timestep" for "target_brightness"', model.coef_[0])

# TODO seasonality ?
# TODO serial dependency dynamic lag plots
# TODO serial dependency autocorrelation plot and what it suggests!

st.divider()
st.markdown('''
## 2 Data Preparation
''')
st.markdown('''
### 2.1 Merging Dataframes + Imputation
We want to join dataframes so we can use inlet_brightness like the other features.  
We can simply impute the mean values for ``inlet_brightness`` since we haven't found any trends or patterns in the data exploration. 
''')
with st.echo():
    merged = pd.merge(inp1, inp2, on='date', how='outer')
    merged['inlet_brightness'] = merged['inlet_brightness'].interpolate(method='linear')

st.dataframe(merged.head())

st.markdown('''
### 2.2 Downsampling
Unfortunately the measurements for our target variable ``target_brightness`` are the finest granulated value we can go down to without imputing the target variable. However as a baseline we will downsample by only allowing values of the merged dataset where we have measurements of our target variable.  
alternatives:
- aggregate values between measurements
- impute target variable
''')
with st.echo():
    downsampled = pd.merge(merged, targ, on='date', how='inner')

st.dataframe(downsampled.head())

# TODO optional aggregate between measurements
# TODO impute target variable? does that make sense?

st.markdown('''
### 2.3 Lags / Steps
- __Lead Time__ : 2h = 1 lag
- __Forecast Horizon__ : 2h = 1 step

this means we will look 1 step backwards and forwards for a prediction
we also need to handle NaN values caused by the shifting
''')
# TODO make num_lags and num_steps dynamic!
# TODO option to fill or impute instead of dropna!
with st.echo():
    num_lags = 1 # create this many features from the past
    for i in range(1, num_lags + 1):
        downsampled[f'y_lag_{i}'] = downsampled['target_brightness'].shift(i)
    
    num_steps = 1 # create this many targets from the future
    for i in range(1, num_steps+ 1):
        downsampled[f'y_step_{i}'] = downsampled['target_brightness'].shift(-i)

    # drop NaN value lines
    downsampled.dropna(inplace=True)

st.dataframe(downsampled.head())

st.markdown('''
### 2.4 Create Train/Test Splits
''')
with st.echo():
    target_columns = ['target_brightness'] + [col for col in downsampled.columns if col.startswith('y_step_')]
    feature_columns = [x for x in list(downsampled.columns) if x not in target_columns]
    feature_columns.remove('date')
    X = downsampled[feature_columns]
    y = downsampled[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

st.markdown('#### input features')
st.dataframe(X.head())
st.markdown('#### target features')
st.dataframe(y.head())

st.markdown('''
# 3. Model Training and Evaluation
''')
with st.echo():

    def evaluate(y_train :pd.DataFrame, y_fit :pd.DataFrame, y_pred :pd.DataFrame) -> dict:
        # root mean squared error
        train_rmse = mean_squared_error(y_train, y_fit, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)

        # r squared
        train_r2 = r2_score(y_train, y_fit)
        test_r2 = r2_score(y_test, y_pred)
        train_r2_raw = r2_score(y_train, y_fit, multioutput='raw_values')
        test_r2_raw = r2_score(y_test, y_pred, multioutput='raw_values')

        return {
            'train_rmse' : train_rmse, 
            'test_rmse' : test_rmse, 
            'train_r2' : train_r2, 
            'test_r2' : test_r2, 
            'train_r2_forecast' : train_r2_raw[1], 
            'test_r2_forecast' : test_r2_raw[1]
        }

# TODO think about what the delta in the metric displays stand for... (vs baseline or vs fit?)

st.markdown('''
## 3.1 Linear Regression (Baseline)
''')

with st.echo():
    linear = LinearRegression() # linear regression as baseline
    # model = MultiOutputRegressor(GradientBoostingRegressor(random_state=1)) alternative
    linear .fit(X_train, y_train)
    y_fit = pd.DataFrame(linear.predict(X_train), index=X_train.index, columns=targ_features)
    y_pred = pd.DataFrame(linear.predict(X_test), index=X_test.index, columns=targ_features)
    base_evalu = evaluate(y_train, y_fit, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (test)", round(base_evalu['test_rmse'],3))
col2.metric("r^2 (test)", round(base_evalu['test_r2'],3))
col3.metric("r^2-forecast (test)", round(base_evalu['test_r2_forecast'],3))

# PLOT COEFFICIENTS
coefficients_df = pd.DataFrame({
    "feature": feature_columns * 2,  # Repeat feature names for both sets
    "coefficient": list(linear.coef_[0]) + list(linear.coef_[1]),
    "target": ["target_brightness"] * len(feature_columns) + ["y_step_1"] * len(feature_columns)
})

# Create a bar chart with different bar colors for each set
fig = px.bar(
    coefficients_df,
    x="feature",
    y="coefficient",
    color="target",
    color_discrete_map={"target_brightness": "gray", "y_step_1": "blue"},  # Set colors for each set
    title="Feature Coefficients",
    barmode='group'
)
# Customize layout if needed
fig.update_layout(
    xaxis_title="Feature",
    yaxis_title="Coefficient Value",
    showlegend=True,  # Set to True to show the legend
)
st.plotly_chart(fig)
st.markdown('''
Note that the coefficients for target_brightness (grey) are irrelevant here, since they try to predict the currently measured paper with the current data from the process.
(The paper that is actually produced with these feature values is actually measured in the following step!)    
The most predictive feature for the target_brightness of a future step (=y_step_1) is soda, closely followed by inlet_brightness and pH value.  
As expected from the data exploration the lag component has rather minute predictive influence on the next timestep.  
If I would be able to make recommendations I'd suggest measuring the inlet brightness every 5 minutes if possible as it is the 2nd most predictive feature - if possible ofc.
''')


st.divider()
st.markdown('''
## 3.2 Neural Network MLP Regressor
''')
with st.echo():
    mlp = MultiOutputRegressor(
        MLPRegressor(
            hidden_layer_sizes=(300, 50, 20),
            max_iter=1000, 
            random_state=42,
            early_stopping=True,
            shuffle=False))

    # Fit and transform the StandardScalers on the training data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Transform the test data using the fitted scalers
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Fit Model
    mlp.fit(X_train_scaled, y_train_scaled)
    y_fit_scaled = pd.DataFrame(mlp.predict(X_train_scaled), index=X_train.index, columns=target_columns)
    y_pred_scaled = pd.DataFrame(mlp.predict(X_test_scaled), index=X_test.index, columns=target_columns)
    y_fit = scaler_y.inverse_transform(y_fit_scaled) # reverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled) # reverse transform predictions
    evalu = evaluate(y_train, y_fit, y_pred)

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (test)", round(evalu['test_rmse'],3), round(evalu['test_rmse'] - base_evalu['test_rmse'], 3))
col2.metric("r^2 (test)", round(evalu['test_r2'],3), round(evalu['test_r2'] - base_evalu['test_r2'], 3))
col3.metric("r^2-forecast (test)", round(evalu['test_r2_forecast'],3), round(base_evalu['test_r2_forecast'] - evalu['train_r2_forecast'], 3))

# TODO ## 3.3 find a good model or at least automl

