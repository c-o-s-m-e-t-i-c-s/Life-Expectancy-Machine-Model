import pandas as pd

from life_expectancy.setup.data_modeling import setup_model
from life_expectancy.display.show_analysis import show_analysis
from life_expectancy.predict.predict_life_expectancy import predict_life_expectancy
from life_expectancy.visualizations.eda import visualizations

def main():
    """Main application loop."""
    print("Initializing Model... (Attempting to load data from 'final_data.csv'...)")
    # Train the model once at startup
    model_pipeline, analysis_df = setup_model()

    while True:
        print("\n" + "=" * 50)
        print("          LIFE EXPECTANCY MODEL DEMO")
        print("=" * 50)
        print("1. View Predictive Analysis (Coefficients)")
        print("2. Predict Life Expectancy from Input Features")
        print("3. View Plots")
        print("4. Exit")
        print("=" * 50)

        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == '1':
            show_analysis(analysis_df)
        elif choice == '2':
            predict_life_expectancy(model_pipeline)
        elif choice == '3':
            visualizations()
        elif choice == '4':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    # Suppress scientific notation for cleaner display of coefficients
    pd.options.display.float_format = '{:.4f}'.format
    main()
