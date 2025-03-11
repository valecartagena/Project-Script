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

#Make school_day and holiday into 1 for yes and 0 for no
dataset['school_day'] = dataset['school_day'].replace({'yes': 1, 'no': 0})
dataset['holiday'] = dataset['holiday'].replace({'yes': 1, 'no': 0})

# Select features (excluding target variable)
x_train_name = dataset.columns.drop(["demand", "date","RRP","RRP_positive","RRP_negative","frac_at_neg_RRP","min_temperature","max_temperature","solar_exposure","rainfall","school_day","holiday"])
x_train_data = dataset[x_train_name].apply(pd.to_numeric, errors='coerce').fillna(0).values

# Train-test split with 80% train, 20% test
x_train, x_test, y_train, y_test = train_test_split(x_train_data, dataset["demand"], test_size=0.2, random_state=42)

# Create linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions
demand_pred = model.predict(x_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, demand_pred))
r2 = r2_score(y_test, demand_pred)
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")

# Print the beta coefficients (model weights) and the intercept
print("Intercept (Beta_0):", model.intercept_)
print("Coefficients (Beta_i for each feature):")

for feature, coef in zip(x_train_name, model.coef_):
    print(f"{feature}: {coef}")

    
# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, demand_pred, alpha=0.6)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Energy Demand")
plt.show()

# # Make predictions
# predictions = demand_pred

# # Determine the threshold using median of training data
# threshold = np.median(y_train)  # Find the median demand in the training set

# # Convert y_train and y_test into binary categories
# y_train_binary = (y_train > threshold).astype(int)  # 1 if above median, else 0
# y_test_binary = (y_test > threshold).astype(int)    # 1 if above median, else 0


# # Compute AUROC
# def calculate_auroc(preds, labels):
#     return roc_auc_score(labels, preds)

# # Permutation test function
# def permutation_test(predictions, y_test_binary, n_permutations=1000):
#     observed_auroc = calculate_auroc(predictions, y_test_binary)
#     null_distribution = np.zeros(n_permutations)

#    # np.random.seed(42)  # Ensure reproducibility
#     for i in range(n_permutations):
#         shuffled_labels = np.random.permutation(y_test_binary)  # Shuffle labels
#         null_distribution[i] = calculate_auroc(predictions, shuffled_labels)  # Compute AUROC

#     # Compute p-value: proportion of permuted AUROCs >= observed AUROC
#     p_value = np.mean(null_distribution >= observed_auroc)
    
#     return null_distribution, p_value, observed_auroc

# # Running permutation test
# n_permutations = 1000
# null_distribution, test_p_value, test_observed_auroc = permutation_test(predictions, y_test_binary, n_permutations)

# print(f"Permutation Test p-value: {test_p_value}")