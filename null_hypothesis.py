import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
dataset = pd.read_csv("dataset.csv")

# Convert categorical variables to binary
dataset['school_day'] = dataset['school_day'].replace({'yes': 1, 'no': 0})
dataset['holiday'] = dataset['holiday'].replace({'yes': 1, 'no': 0})

# Select features (excluding target variable)
x_train_name = dataset.columns.drop("demand")
x_train_data = dataset[x_train_name].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Train-test split (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x_train_data, dataset["demand"], test_size=0.2, random_state=42)

# Fit initial linear regression model
model = LinearRegression()
model.fit(x_train, y_train)
original_coefs = model.coef_

# Bootstrap function for hypothesis testing
def bootstrap_test(x, y, n_bootstrap=1000, alpha=0.05):
    np.random.seed(42)  # Ensure reproducibility
    coef_samples = np.zeros((n_bootstrap, x.shape[1]))  # Store coefficients

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(len(y), len(y), replace=True)
        x_resampled, y_resampled = x[indices], y.iloc[indices]

        # Fit model on bootstrap sample
        model = LinearRegression()
        model.fit(x_resampled, y_resampled)

        # Store coefficients
        coef_samples[i] = model.coef_

    # Compute confidence intervals
    lower_bound = np.percentile(coef_samples, 100 * (alpha / 2), axis=0)
    upper_bound = np.percentile(coef_samples, 100 * (1 - alpha / 2), axis=0)

    return coef_samples, lower_bound, upper_bound

# Run bootstrap hypothesis test
n_bootstrap = 100000
alpha = 0.05  # 95% confidence interval
bootstrap_coefs, lower_bound, upper_bound = bootstrap_test(x_train, y_train, n_bootstrap, alpha)

# Print results with hypothesis testing
print("\nBootstrap Hypothesis Test (95% Confidence Intervals):")
for feature, coef, lower, upper in zip(x_train_name, original_coefs, lower_bound, upper_bound):
    significant = "Reject H0 (Significant)" if lower > 0 or upper < 0 else "Fail to Reject H0 (Not Significant)"
    print(f"{feature}: {coef:.4f}  (95% CI: [{lower:.4f}, {upper:.4f}]) → {significant}")
