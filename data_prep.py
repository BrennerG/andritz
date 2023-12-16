import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go


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
Unfortunately the measurements for our target variable ``target_brightness`` are the finest granulated value we can go down to without imputing the target variable. However as a baseline we will downsample by only allowing values of the merged dataset where we have measurements of our target variable.  
alternatives:
- aggregate values between measurements
- impute target variable
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
we also need to handle NaN values caused by the shifting
''')
with st.echo():

    num_lags = 1 # create this many features from the past
    for i in range(1, num_lags + 1):
        downsampled[f'y_lag_{i}'] = downsampled['target_brightness'].shift(i)
    
    num_steps = 1 # create this many targets from the future
    for i in range(1, num_steps+ 1):
        downsampled[f'y_step_{i}'] = downsampled['target_brightness'].shift(-i)

    # drop NaN value lines
    downsampled.dropna(inplace=True)

    st.dataframe(downsampled)


st.markdown('''
### Create Train/Test Splits
''')
with st.echo():
    target_columns = ['target_brightness'] + [col for col in downsampled.columns if col.startswith('y_step_')]
    feature_columns = [x for x in list(downsampled.columns) if x not in target_columns]
    feature_columns.remove('date')
    X = downsampled[feature_columns]
    y = downsampled[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

F"features: {feature_columns}"
F"targets: {target_columns}"


st.markdown('''
### Baseline Model
Linear Regression
''')
with st.echo():
    model = LinearRegression() # linear regression as baseline
    model.fit(X_train, y_train)
    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y.columns)

    # root mean squared error
    train_rmse = mean_squared_error(y_train, y_fit, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    # r squared
    train_r2 = r2_score(y_train, y_fit)
    test_r2 = r2_score(y_test, y_pred)
    train_r2_raw = r2_score(y_train, y_fit, multioutput='raw_values')
    test_r2_raw = r2_score(y_test, y_pred, multioutput='raw_values')
    
F"Train RMSE: {train_rmse:.2f}\n" f"Test RMSE: {test_rmse:.2f}"
F"Train R^2: {train_r2:.2f}\n" f"Test R^2: {test_r2:.2f}"
F"Train Raw R^2: {train_r2_raw}\n" f"Test Raw R^2: {test_r2_raw}"
#eval_data = {'metric': ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'train_r2_forecast', 'test_r2_forecast'],  'value': [train_rmse, test_rmse, train_r2, test_r2, train_r2_raw[1], test_r2_raw[1]]}
#fig = go.Figure()
#fig.add_trace(go.Bar(x=eval_data['value'], y=eval_data['metric'], orientation='h', marker=dict(color='blue')))
# fig.update_layout(title='Baseline Evaluation', xaxis_title='values', yaxis_title='metrics')
# st.plotly_chart(fig)

st.markdown('''
### Analysis
''')
with st.echo():
    coefficients = model.coef_
    intercept = model.intercept_

    coefficients_df = pd.DataFrame({
        "feature": feature_columns * 2,  # Repeat feature names for both sets
        "coefficient": list(coefficients[0]) + list(coefficients[1]),
        "target": ["target_brightness"] * len(feature_columns) + ["y_step_1"] * len(feature_columns)
    })

# coefficients_df

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