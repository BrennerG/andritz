import streamlit as st
import pandas as pd
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


st.title('Model Selection')


with st.echo():
    # Load split indices from the file
    with open('data/train_test_split_indices.pkl', 'rb') as file:
        loaded_split_indices = pickle.load(file)

    # Retrieve the split indices
    X_cols = loaded_split_indices['X_cols']
    y_cols = loaded_split_indices['y_cols']
    X_train = loaded_split_indices['X_train']
    X_test = loaded_split_indices['X_test']
    y_train = loaded_split_indices['y_train']
    y_test = loaded_split_indices['y_test']

    def evaluate(y_train :pd.DataFrame, y_fit :pd.DataFrame, y_pred :pd.DataFrame) -> dict:
        # root mean squared error
        train_rmse = mean_squared_error(y_train, y_fit, squared=False)
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)

        # r squared
        train_r2 = r2_score(y_train, y_fit)
        test_r2 = r2_score(y_test, y_pred)
        train_r2_raw = r2_score(y_train, y_fit, multioutput='raw_values')
        test_r2_raw = r2_score(y_test, y_pred, multioutput='raw_values')
        return  {
            'metric': ['train_rmse', 'test_rmse', 'train_r2', 'test_r2', 'train_r2_forecast', 'test_r2_forecast'],
            'value': [train_rmse, test_rmse, train_r2, test_r2, train_r2_raw[1], test_r2_raw[1]]}
    

st.markdown('''
### Gradient Boosting Regressor
''')
with st.echo():
    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=1))
    model.fit(X_train, y_train)
    y_fit = pd.DataFrame(model.predict(X_train), index=X_train.index, columns=y_cols)
    y_pred = pd.DataFrame(model.predict(X_test), index=X_test.index, columns=y_cols)
    st.write(evaluate(y_train, y_fit, y_pred))


st.markdown('''
### Neural Network MLP Regressor
''')
with st.echo():
    model = MultiOutputRegressor(
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
    model.fit(X_train_scaled, y_train_scaled)
    y_fit_scaled = pd.DataFrame(model.predict(X_train_scaled), index=X_train.index, columns=y_cols)
    y_pred_scaled = pd.DataFrame(model.predict(X_test_scaled), index=X_test.index, columns=y_cols)
    y_fit = scaler_y.inverse_transform(y_fit_scaled) # reverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled) # reverse transform predictions
    st.write(evaluate(y_train, y_fit, y_pred))