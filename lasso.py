import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

# Load data
dataset = pd.read_csv("dataset.csv")

# Convert date column to datetime format
dataset['date'] = pd.to_datetime(dataset['date'])

# Make school_day and holiday into 1 for yes and 0 for no
dataset['school_day'] = dataset['school_day'].replace({'yes': 1, 'no': 0})
dataset['holiday'] = dataset['holiday'].replace({'yes': 1, 'no': 0})

# Select features (excluding target variable)
x_train_name = dataset.columns.drop("demand")
x_train_data = dataset[x_train_name].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Train-test split with 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x_train_data, dataset["demand"], test_size=0.2, random_state=42)

# Create lasso regression model
Lasso_Regression = Lasso(alpha=0.9)
Lasso_Regression.fit(x_train, y_train)

# Print R^2 score of the model
train_score = Lasso_Regression.score(x_train, y_train)
print(f"Lasso Regression R^2 Score on Training Data: {train_score}")

# Make predictions
demand_pred = Lasso_Regression.predict(x_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, demand_pred))
r2 = r2_score(y_test, demand_pred)
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, demand_pred, alpha=0.6)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Energy Demand Lasso Regression")
plt.show()

# Get coefficients (beta values) from the trained model
coefficients = Lasso_Regression.coef_
print("Lasso Regression Coefficients:")

# Print the feature names and their corresponding coefficients
for feature, coef in zip(x_train_name, coefficients):
    print(f"{feature}: {coef}")

# Identify the important features (non-zero coefficients)
important_features = zip(x_train_name, coefficients)
important_features = [f"{feature}: {coef}" for feature, coef in important_features if coef != 0]

print("\nImportant Features (non-zero coefficients):")
for feature in important_features:
    print(feature)
