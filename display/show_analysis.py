import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import sys

def show_analysis(analysis_df):
    """Displays the key model coefficients from the MLR analysis."""
    print("\n" + "=" * 50)
    print("      MULTIPLE LINEAR REGRESSION COEFFICIENTS")
    print("=" * 50)
    print("Impact on Life Expectancy (Standardized Coefficients):")
    print("-----------------------------------------------------")
    print("NOTE: Coefficient represents the change in Life Expectancy (years)")
    print("      for a 1-Standard Deviation increase in the feature.\n")

    # 1. LOAD ACTUAL DATA (Requires 'final_data.csv' to be in the execution directory)
    try:
        # This assumes your final, merged dataset is saved as 'final_data.csv'.
        # The file MUST contain all columns used for training.
        df = pd.read_csv('final_data.csv', index_col=0)
        print("Successfully loaded data from 'final_data.csv'.")
    except FileNotFoundError:
        print("\n\n" + "#" * 70)
        print("FATAL ERROR: Data file 'final_data.csv' not found.")
        print("To use your actual data, you must save your final merged DataFrame")
        print("as 'final_data.csv' in the same directory as this script.")
        print("The application will now exit.")
        print("#" * 70 + "\n")
        sys.exit(1)

    #copy pasting code from prev data_modeling.py
    y = df['Life Expectancy']
    X = df.drop('Life Expectancy', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

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

    # --- 1. Evaluate Model Performance (R-squared) ---
    r2_score_test = model_pipeline.score(X_test, y_test)
    print(f"\nModel Performance (R-squared on Test Data): {r2_score_test:.4f}")

    # --- 2. Extract and Interpret Coefficients ---

    # Get feature names after preprocessing
    feature_names_out = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
    # Get the coefficients from the final LinearRegression step
    coefficients = model_pipeline.named_steps['regressor'].coef_

    # Create a DataFrame for clear interpretation
    coef_df = pd.DataFrame({
        'Feature': feature_names_out,
        'Coefficient': coefficients
    })

    # Filter to show only the main numerical features (those starting with 'num__')
    # and sort them by the magnitude (absolute value) of their impact
    main_features_impact = coef_df[coef_df['Feature'].str.startswith('num__')].copy()

    # Clean up feature names for display
    main_features_impact['Feature'] = main_features_impact['Feature'].str.replace('num__', '')
    main_features_impact['Abs_Coefficient'] = main_features_impact['Coefficient'].abs()

    # Display the main numerical features sorted by absolute coefficient magnitude
    print("\n--- Model Coefficients: Predictive Impact on Life Expectancy ---")
    print(
        "NOTE: These coefficients represent the change in Life Expectancy (in years) \n      for a 1-Standard Deviation increase in the predictor variable.")
    print(main_features_impact.sort_values(by='Abs_Coefficient', ascending=False).drop(columns='Abs_Coefficient'))