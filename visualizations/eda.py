import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import sys

def visualizations():
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

    while True:
        print("\n" + "=" * 50)
        print("          LIFE EXPECTANCY MODEL DEMO")
        print("=" * 50)
        print("1. Descriptive Statistics")
        print("2. Histogram Plots")
        print("3. Scatterplots")
        print("4. Correlation Heatmap")
        print("5. Prediction Results")
        print("6. Back")
        print("=" * 50)

        choice = input("Enter your choice (1, 2, 3, 4, 5, or 6): ").strip()

        if choice == '1':
            descriptive(df)
        elif choice == '2':
            histogram(df)
        elif choice == '3':
            scatter(df)
        elif choice == '4':
            correlation(df)
        elif choice == '5':
            prediction(df)
        elif choice == '6':
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5 or 6.")

def descriptive(df):
    numerical_cols = ['Year', 'Life Expectancy', 'Health Exp %', 'DPT3 Vac %', 'PSI', 'Rural Pop %']

    for col in numerical_cols:
        print(f"\n--- Feature: {col} ---")

        summary = df[col].describe()
        print("\nPandas Summary:")
        print(summary)

        skewness = df[col].skew()
        print(f"Skewness: {skewness:.4f}")

        kurtosis = df[col].kurt()
        print(f"Kurtosis: {kurtosis:.4f}")

        print("-" * 30)

def histogram(df):
    numerical_cols = ['Year', 'Life Expectancy', 'Health Exp %', 'DPT3 Vac %', 'PSI', 'Rural Pop %']
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['figure.dpi'] = 100

    for col in numerical_cols:
        plt.figure()

        sns.histplot(df[col], kde=True, bins=20, color='skyblue')
        plt.title(f'Histogram: Distribution of {col}', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        plt.tight_layout()
        plt.show()

def scatter(df):
    columns = ['DPT3 Vac %', 'Health Exp %', 'PSI', 'Rural Pop %']

    while True:
        print("1. DPT 3 %")
        print("2. Health Exp %")
        print("3. PSI")
        print("4. Rural Pop %")
        scatterinp = input('Enter your choice (1, 2, 3, or 4): ').strip()
        if scatterinp == '1':
            inp = columns[0]
            break
        elif scatterinp == '2':
            inp = columns[1]
            break
        elif scatterinp == '3':
            inp = columns[2]
            break
        elif scatterinp == '4':
            inp = columns[3]
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    plt.figure(figsize=(8, 6))

    sns.scatterplot(x=f'{inp}', y='Life Expectancy', color='darkred', alpha=0.3, data=df)
    plt.title(f'Relationship between {inp} and Life Expectancy', fontsize=16)
    plt.show()

def correlation(df):
    numerical_cols = ['Year', 'Life Expectancy', 'Health Exp %', 'DPT3 Vac %', 'PSI', 'Rural Pop %']
    correlation_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5
    )

    plt.title('Correlation Matrix of Numerical Features', fontsize=16)
    plt.show()

def prediction(df):
    # reuses code from model building

    # Define Features and Target
    y = df['Life Expectancy']
    X = df.drop('Life Expectancy', axis=1)

    # Define categorical and numerical columns
    categorical_features = ['Country']
    numerical_features = ['Year', 'Health Exp %', 'DPT3 Vac %', 'PSI', 'Rural Pop %']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    lr = LinearRegression()
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


    plt.figure(figsize=(5, 5))
    plt.scatter(x=y_train, y=y_lr_train_pred, c='seagreen', alpha=0.3)

    z = np.polyfit(y_train, y_lr_train_pred, 1)
    p = np.poly1d(z)

    plt.ylabel('Predicted Life Expectancy')
    plt.xlabel('Experimental Life Expectancy')

    plt.plot(y_train, p(y_train), 'red')
    plt.show()