from typing import Tuple

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



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
# TODO add seasonal plot
# TODO deploy...
# ---
# OPTIONALs
# TODO 3.3 find a good model or at least automl
# TODO show more in explo! what to talk about?
# ---
# IDEAS and Thoughts
# - why is the r2 of lag2 step2 so much better than lag1 step1?
# - but why does it latch on the lag arguments instantly?
# ---
# FUTURE WORK
# - Feature selection and Feature Mutation
# - cool neural model
# - talk to an expert lol

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
st.dataframe(inp1)
st.markdown('#### Input 2')
st.dataframe(inp2)
st.markdown('#### Target')
st.dataframe(targ)

st.markdown('#### NaN Values')
col1, col2, col3 = st.columns(3)
col1.markdown('inp1')
col1.write(inp1.isna().sum())
col2.markdown('inp2')
col2.write(inp2.isna().sum())
col3.markdown('target')
col3.write(targ.isna().sum())

st.markdown('''
#### Observation
There is a time discrepancy between the measurements of the different features
- inp1 are measured every 5 minutes
- inp2 is measured every 2 hours
- target is measured every 2 hours with a 3h45min delay
''')

st.markdown('''
## 1.1 Stationarity
The following function plots a given feature over time aggregating data points via a moving average of a given window size.
window size of 1 = original data, 288 = daily average (for inp1)
The light blue line is a linear regression model fit on the aggregated data that allows us to make conclusions about time dependency of a given feature - it's coefficient is shown below the plot.  
''')
with st.echo(): 
    def moving_average_plot(feature_set :pd.DataFrame, selected_feature :str, window :int) -> Tuple[go.Figure, float]:
        feature_set = feature_set.copy()
        feature_set['step'] = np.arange(len(feature_set.index))

        # create moving average
        moving_average = feature_set[selected_feature].rolling(
            window=window,      
            center=True,        
            min_periods=round(window/2),     
        ).mean()
        feature_set['moving_avg_'+selected_feature] = moving_average

        # linear regression
        X =feature_set.loc[:, ['step']]
        y =feature_set.loc[:, 'moving_avg_'+selected_feature]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=X.index, name='lin_reg')
        linreg_viz_df = pd.merge(feature_set, y_pred.to_frame(), left_index=True, right_index=True)
        fig = px.line(linreg_viz_df, x='date', y=['moving_avg_'+selected_feature, 'lin_reg'])
        fig.update_layout(title=f'Moving Average for {selected_feature}', xaxis_title='Date', yaxis_title=selected_feature)
        return fig, model.coef_[0]

st.markdown('### Feature Set 1')
selected_feature = st.selectbox("Moving Average for:", [x for x in inp1_features if x != 'date'], index=0,)
window = st.slider("moving average (window)", 1, 3000, 288)
fig, coef = moving_average_plot(inp1, selected_feature=selected_feature, window=window)
st.plotly_chart(fig)
st.metric(F'coefficient: "timestep" for "{selected_feature}"', coef)

st.divider()
st.markdown('### Feature Set 2 (inlet_brightness)')
window2 = st.slider("feature set 2: moving average (window)", 1, 100, 6)
fig, coef = moving_average_plot(inp2, selected_feature="inlet_brightness", window=window2)
st.plotly_chart(fig)
st.metric(F'coefficient: "timestep" for inlet_brightness', coef)

st.divider()
st.markdown('### Target Variable (target_brightness)')
window3 = st.slider("target: moving average (window)", 1, 100, 6)
fig, coef = moving_average_plot(targ, selected_feature='target_brightness', window=window3)
st.plotly_chart(fig)
st.metric(F'coefficient: "timestep" for target_brightness', coef)

st.markdown('''
We cannot observe any time dependent trends in the featuresets nor the target variable, so we don't have to explicitly model them later during feature engineering.
''')

st.divider()
st.markdown('''
### Autocorrelation with lags (target variable)
The following plot shows the correlation of the target variable with its previous measurements.
It suggests significant correlation up until 8 steps backward ("lags").
''')
lags = range(1, 30)  # Choose the number of lags to include in the plot
autocorrelation_values = [targ['target_brightness'].autocorr(lag=lag) for lag in lags]
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(lags), y=autocorrelation_values, mode='markers+lines', name='Autocorrelation'))
fig.update_layout(title='Autocorrelation Plot', xaxis_title='Lag', yaxis_title='Autocorrelation')
st.plotly_chart(fig)


st.markdown('''
## 2 Data Preparation
''')
st.markdown('''
### 2.1 Merging Dataframes + Imputation
Firstly we'll join dataframes so we can use inlet_brightness like the other features.  
We can simply impute the mean values for ``inlet_brightness`` using linear interpolation.
(In a real scenario I'd consider an experts opinion for the imputation of any value)
''')
with st.echo():
    merged = pd.merge(inp1, inp2, on='date', how='outer')
    merged['inlet_brightness'] = merged['inlet_brightness'].interpolate(method='linear')

st.dataframe(merged.head())

st.markdown('''
### 2.2 Aggregation
Due to the time discrepancy in the measurement of our target variable we are left with either imputing the target variable or dropping all rows without a target measurement.  
Since imputing the target variable itself is a dangerous process, in my opinion downsampling is the correct way of proceeding, however I also implemented to other approaches for experimentation:  
(evaluation will still be done using the non-imputed data)
''')

method = st.radio(
    "Aggregation Method",
    ["Downsampling", "Fill", "Interpolation"],
    captions = [
        "Drop all feature rows with NaN target values", 
        "Fill NaN target values forward",
        "Linear Interpolation of target values"])

with st.echo():
    if method == "Downsampling":
        data = pd.merge(merged, targ, on='date', how='inner')

    elif method == "Fill":
        data = pd.merge(merged, targ, on='date', how='outer')
        data['target_brightness'].fillna(method='ffill', inplace=True)
        downsampled = pd.merge(merged, targ, on='date', how='inner')

    elif method == "Interpolation":
        data = pd.merge(merged, targ, on='date', how='outer')
        data['target_brightness'] = data['target_brightness'].interpolate(method='linear')
        downsampled = pd.merge(merged, targ, on='date', how='inner')

st.dataframe(data)


st.markdown('''
### 2.3 Lags / Steps
Lags enable us to consider the previous X target measure during the prediction step.  
Steps are the next X measurements in the future that are required to be predicted.  
Picking parameters for them is dependent of the actual use case, but since the requirements were not clear lags and steps can be set dynamically using the sliders below.
''')
with st.echo():
    def make_lags_and_steps(data :pd.DataFrame, num_lags :int, num_steps :int) -> pd.DataFrame:
        for i in range(1, num_lags + 1): # create this many features from the past
            data[f'y_lag_{i}'] = data['target_brightness'].shift(i)
        
        for i in range(1, num_steps+ 1): # create this many targets from the future
            data[f'y_step_{i}'] = data['target_brightness'].shift(-i)

        # drop NaN value lines
        data = data.dropna()

        return data

num_lags = st.slider('lag', 1, 10, 2)
num_steps = st.slider('steps', 1, 10, 2)
with st.echo(): 
    data = make_lags_and_steps(data, num_lags, num_steps)
st.dataframe(data.head())

st.markdown('''
### 2.4 Create Train/Test Splits
Creating a train/test split without shuffling.  
The last quarter of the recorded data is used for evaluation purposes.
''')
with st.echo():
    target_columns = ['target_brightness'] + [col for col in data.columns if col.startswith('y_step_')]
    feature_columns = [x for x in list(data.columns) if x not in target_columns]
    feature_columns.remove('date')
    X = data[feature_columns]
    y = data[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

st.markdown(F'#### input features {X.shape}')
st.dataframe(X.head())
st.markdown(F'#### target features {y.shape}')
st.dataframe(y.head())


st.markdown('''
# 3. Model Training and Evaluation
Root Mean Squared Error, R Squared and Mean Absolute Error are considered for evaluation.  
The values shown for these metrics are the mean for all target variables (forecasting steps)
''')
with st.echo():

    def evaluate(y_fit :pd.DataFrame, y_pred :pd.DataFrame,  y_train :pd.DataFrame, y_test:pd.DataFrame) -> dict:
        # root mean squared error
        train_rmse = mean_squared_error(y_train, y_fit, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        train_rmse_raw = mean_squared_error(y_train, y_fit, squared=False, multioutput='raw_values')
        test_rmse_raw = mean_squared_error(y_test, y_pred, squared=False, multioutput='raw_values')

        # r squared
        train_r2 = r2_score(y_train, y_fit)
        test_r2 = r2_score(y_test, y_pred)
        train_r2_raw = r2_score(y_train, y_fit, multioutput='raw_values')
        test_r2_raw = r2_score(y_test, y_pred, multioutput='raw_values')

        # mean absolute error
        train_mae = mean_absolute_error(y_train, y_fit)
        test_mae = mean_absolute_error(y_test, y_pred)
        train_mae_raw = mean_absolute_error(y_train, y_fit, multioutput='raw_values')
        test_mae_raw = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

        return {
            'train_rmse' : train_rmse, 
            'test_rmse' : test_rmse, 
            'train_rmse_raw' : train_rmse_raw,
            'test_rmse_raw' : test_rmse_raw,
            'train_r2' : train_r2, 
            'test_r2' : test_r2, 
            'train_r2_raw' : train_r2_raw, 
            'test_r2_raw' : test_r2_raw,
            'train_mae' : train_mae,
            'test_mae' : test_mae,
            'train_mae_raw' : train_mae_raw,
            'test_mae_raw' : test_mae_raw
        }
    
def evaluate_filled(downsampled :pd.DataFrame, model):
    downsampled = make_lags_and_steps(downsampled, num_lags, num_steps)

    target_columns = ['target_brightness'] + [col for col in downsampled.columns if col.startswith('y_step_')]
    feature_columns = [x for x in list( downsampled.columns) if x not in target_columns]
    feature_columns.remove('date')
    X = downsampled[feature_columns]
    y = downsampled[target_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=target_columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=target_columns)

    return evaluate(y_fit, y_pred, y_train, y_test)

def data_vs_prediction_plot(data :pd.DataFrame, target_columns :list, y_fit :pd.DataFrame, y_pred :pd.DataFrame) -> go.Figure:
    trace1 = go.Scatter(x=data['date'], y=data['target_brightness'], mode='lines', name='original data')
    traces = [trace1]
    for target in target_columns:
        traces.append(go.Scatter(x=data['date'], y=y_fit[target], mode='lines', name=target+'_fit'))
        traces.append(go.Scatter(x=data.tail(y_pred.shape[0])['date'], y=y_pred[target], mode='lines', name=target+'_pred'))

    layout = go.Layout(title='Original Data vs Prediction', xaxis=dict(title='date'), yaxis=dict(title='target_brightness'))
    fig = go.Figure(data=traces, layout=layout)
    return fig

RAW_METRICS_DIC = {
    'RMSE': ['train_rmse_raw', 'test_rmse_raw'],
    'MAE': ['train_mae_raw', 'test_mae_raw'],
    'R^2': ['train_r2_raw', 'test_r2_raw']
}

def metrics_over_forecast_plot(selected_metric:str, evaluation_dic:dict):
    traces = []
    for sel in RAW_METRICS_DIC[selected_metric]:
        traces.append(go.Scatter(x=np.arange(0,len(evaluation_dic[sel])), y=evaluation_dic[sel], mode='lines', name=sel))
    layout = go.Layout(title=f'Forecast Horizon: {selected_metric}', xaxis=dict(title='steps'), yaxis=dict(title=selected_metric))
    fig = go.Figure(data=traces, layout=layout)
    return fig


st.markdown('''
## 3.1 Linear Regression (Baseline)
''')

with st.echo():
    linear = LinearRegression() # linear regression as baseline
    linear.fit(X_train, y_train)
    y_fit = pd.DataFrame(linear.predict(X_train), index=X_train.index, columns=target_columns)
    y_pred = pd.DataFrame(linear.predict(X_test), index=X_test.index, columns=target_columns)
    base_evalu = evaluate(y_fit, y_pred, y_train, y_test) if method=="Downsampling" else evaluate_filled(downsampled, linear)

# Original Data vs Prediction
st.plotly_chart(data_vs_prediction_plot(data, target_columns,  y_fit ,y_pred))

# show metrics
st.markdown('#### Evaluation Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("RMSE (train)", round(base_evalu['train_rmse'],3))
col1.metric("RMSE (test)", round(base_evalu['test_rmse'],3))
col2.metric("MAE (train)", round(base_evalu['train_mae'],3))
col2.metric("MAE (test)", round(base_evalu['test_mae'],3))
col3.metric("r^2 (train)", round(base_evalu['train_r2'],3))
col3.metric("r^2 (test)", round(base_evalu['test_r2'],3))

selected_metric = st.selectbox("plot metric over forecast horizon", RAW_METRICS_DIC.keys(), index=2)
st.plotly_chart(metrics_over_forecast_plot(selected_metric, base_evalu))
st.markdown('''
This plot allows us to see the evaluated metrics for the different target variables (=forecasting steps).  
It allows us to see how well the model predicts on any step into the future.  
Depending on the use case it might be more important to have the model be more accurate certain steps ahead.  
Keep in mind that the values shown above are the mean values for all forecasting steps - in this particular setting the R^2 actually increases over the forecast horizon.
''')

st.markdown('#### Model Coefficients')
# plot coefficients
coefficients_df = pd.DataFrame({
    "feature": feature_columns * len(target_columns),  # Repeat feature names for both sets
    "coefficient": np.ravel(linear.coef_),
    # "target": repeat_elements(target_columns, len(feature_columns))
    "target": [item for item in target_columns for _ in range(len(feature_columns))]
})
fig = px.bar( 
    coefficients_df,
    x="feature",
    y="coefficient",
    color="target",
    color_discrete_map={"target_brightness": "gray", "y_step_1": "blue"},  # Set colors for each set
    title="Feature Coefficients",
    barmode='group'
)
fig.update_layout(
    xaxis_title="Feature",
    yaxis_title="Coefficient Value",
    showlegend=True,  # Set to True to show the legend
)
st.plotly_chart(fig) 

st.markdown('''
This plot shows the model's coefficients for the target variables (=forecasting steps) - these can be interpreted as predictive power of a feature to predict a target variable.  
Notice that depending on the forecasting step other features change importance:  
Soda, inlet_brightness and ph-value are important for the prediction of the next measurement step.  
However inlet_brightness, peroxide and soda are more important for the next.  
Longer forecasting horizons seem to favor ph-value and temperature as predictive features moreso than earlier steps in the horizon.
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
    evalu = evaluate(y_fit, y_pred, y_train, y_test) if method=="Downsampling" else evaluate_filled(downsampled, mlp)

# Original Data vs Prediction
st.plotly_chart(data_vs_prediction_plot(data, target_columns, pd.DataFrame(y_fit, columns=target_columns), pd.DataFrame(y_pred, columns=target_columns)))

# show metrics
st.markdown('#### Evaluation Metrics')
col1, col2, col3 = st.columns(3)
col1.metric("RMSE (train)", round(evalu['train_rmse'],3), round(evalu['train_rmse'] - base_evalu['train_rmse'],2))
col1.metric("RMSE (test)", round(evalu['test_rmse'],3), round(evalu['test_rmse'] - base_evalu['test_rmse'],2))
col2.metric("MAE (train)", round(evalu['train_mae'],3), round(evalu['train_mae'] - base_evalu['train_mae'], 2))
col2.metric("MAE (test)", round(evalu['test_mae'],3), round(evalu['test_mae'] - base_evalu['test_mae'],2))
col3.metric("r^2 (train)", round(evalu['train_r2'],3), round(evalu['train_r2'] - base_evalu['train_r2'],2))
col3.metric("r^2 (test)", round(evalu['test_r2'],3), round(evalu['test_r2'] - base_evalu['test_r2'],2))

selected_metric2 = st.selectbox("plot metric over forecast horizon:", RAW_METRICS_DIC.keys(), index=2)
st.plotly_chart(metrics_over_forecast_plot(selected_metric2, evalu))