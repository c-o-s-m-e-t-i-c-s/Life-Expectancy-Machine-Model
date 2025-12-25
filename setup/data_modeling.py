import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import sys

def setup_model():
    """Loads actual data from a file and trains the Multiple Linear Regression model."""

    # 1. LOAD ACTUAL DATA (Requires 'final_data.csv' to be in the execution directory)
    try:
        df = pd.read_csv('final_data.csv', index_col=0)
    except FileNotFoundError:
        print("\n\n" + "#" * 70)
        print("FATAL ERROR: Data file 'final_data.csv' not found.")
        print("To use your actual data, you must save your final merged DataFrame")
        print("as 'final_data.csv' in the same directory as this script.")
        print("The application will now exit.")
        print("#" * 70 + "\n")
        sys.exit(1)

# Ensure all required columns are present before proceeding with modeling
    required_cols = ['Life Expectancy', 'Health Exp %', 'PSI',
                     'Rural Pop %', 'DPT3 Vac %', 'Year', 'Country']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print("\n\n" + "#" * 70)
        print(f"FATAL ERROR: Missing required columns in 'final_data.csv': {', '.join(missing_cols)}")
        print("Please ensure your data file contains all the necessary features.")
        print("#" * 70 + "\n")
        sys.exit(1)

    # ==================
    # DATA PREPARATION
    # ==================

    # Define Features and Target
    y = df['Life Expectancy']
    X = df.drop('Life Expectancy', axis=1)

    ## DATA SPLITTING

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

    # ==================
    # LINEAR REGRESSION
    # TRAINING THE MODEL
    # ==================

    lr = LinearRegression()

    # Define categorical and numerical columns
    categorical_features = ['Country']
    numerical_features = ['Year', 'Health Exp %', 'DPT3 Vac %', 'PSI', 'Rural Pop %']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ],
        remainder='passthrough'
    )

    # Create the Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', lr)  # Using existing LinearRegression model
    ])
    model_pipeline.fit(X_train, y_train)

    y_lr_train_pred = model_pipeline.predict(
        X_train)  # making prediction on the original dataset it has been trained on, evaluate performance
    y_lr_test_pred = model_pipeline.predict(X_test)

    # for training set
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)

    # for testing set
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    lr_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    #print(lr_results)

    # ==================
    # RANDOM FOREST
    # TRAINING THE MODEL
    # ==================

    rf = RandomForestRegressor(max_depth=10, random_state=100)  # increased from 2 to 10 to compensate for underfitting

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])

    rf_pipeline.fit(X_train, y_train)

    # Apply model to make a prediction
    y_rf_train_pred = rf_pipeline.predict(X_train)
    y_rf_test_pred = rf_pipeline.predict(X_test)

    # Evaluate model performance
    # for training set
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)

    # for testing set
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
    #print(rf_results)

    # Model Comparison
    df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(
        drop=True)  # axis=0 to combine in a row-wise manner
    #print(df_models)
    return (model_pipeline, lr_results)