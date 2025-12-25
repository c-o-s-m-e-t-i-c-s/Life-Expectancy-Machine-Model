import pandas as pd

def predict_life_expectancy(model_pipeline):
    """Prompts user for input and provides a Life Expectancy prediction."""
    print("\n" + "=" * 50)
    print("         LIFE EXPECTANCY PREDICTOR TOOL")
    print("=" * 50)

    # Base features to collect from the user
    feature_prompts = {
        'Health Exp %': "Health Expenditure (% GDP, e.g., 6.5): ",
        'PSI': "Political Stability Index (-2.5 to 2.5, e.g., 0.5): ",
        'Rural Pop %': "Rural Population Percentage (e.g., 40): ",
        'DPT3 Vac %': "DPT3 Vaccination Coverage (%, e.g., 90): ",
        'Year': "Year of Data (e.g., 2020): ",
    }

    user_input = {}
    for feature, prompt in feature_prompts.items():
        while True:
            try:
                value = float(input(prompt))
                user_input[feature] = value
                break
            except ValueError:
                print("Invalid input. Please enter a numerical value.")

    # Create the input DataFrame structure required by the pipeline
    # We must include all columns used in training, even if they are fixed for prediction.
    input_data = {
        'Health Exp %': [user_input['Health Exp %']],
        'PSI': [user_input['PSI']],
        'Rural Pop %': [user_input['Rural Pop %']],
        'DPT3 Vac %': [user_input['DPT3 Vac %']],
        'Year': [user_input['Year']],
        'Country': ['A (High LE)'],  # Fix the Country to the baseline for prediction
        'Year_Squared': [user_input['Year'] ** 2]  # Add engineered feature
    }

    X_new = pd.DataFrame(input_data)

    # Ensure column order matches the model training order
    # (The pipeline handles the rest of the feature selection/transformation)
    X_new = X_new[['Health Exp %', 'PSI', 'Rural Pop %', 'DPT3 Vac %', 'Year', 'Country',
                   'Year_Squared']]

    # Make Prediction
    try:
        predicted_le = model_pipeline.predict(X_new)[0]

        print("\n" + "*" * 50)
        print(f"PREDICTED LIFE EXPECTANCY:")
        print(f"Based on your inputs, the predicted Life Expectancy is: {predicted_le:.2f} years")
        print("*" * 50)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")

