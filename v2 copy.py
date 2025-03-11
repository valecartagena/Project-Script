# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split

# #we train our data from 2015-2018, which is 60% of our data
# train_dataset = pd.read_csv("train_dataset.csv")
# #test data on 2019-2020, which is 40% of our data
# test_dataset = pd.read_csv("test_dataset.csv")

# #x_train = train_dataset.iloc[:, "demand"]

# #create our training and testing variables
# y_train = train_dataset.iloc[:, 1]
# y_test = test_dataset.iloc[:, 1]

# # x_name = ['demand','RRP','demand_pos_RRP','RRP_positive','demand_neg_RRP',
# #                 'RRP_negative','frac_at_neg_RRP','min_temperature','max_temperature',
# #                 'solar_exposure','rainfall']
# # x_name = train_dataset.iloc[0]
# x_name = train_dataset.columns
# x_train = np.array(train_dataset[x_name])
# x_test = np.array(test_dataset[x_name])

# #create linear regression
# model = LinearRegression()
# model.fit(x_train, y_train)
# demand_pred = model.predict(x_test)

# # Calculate RMSE and R^2 for evaluation
# rmse = np.sqrt(mean_squared_error(y_test, demand_pred))
# r2 = r2_score(y_test, demand_pred)
# print(f"Root Mean Squared Error: {rmse}")
# print(f"R^2 Score: {r2}")

# #Plot the predictions vs actual values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, demand_pred)
# plt.xlabel("Actual Demand")
# plt.ylabel("Predicted Demand")
# plt.title("Actual vs Predicted Energy Demand")
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
train_dataset = pd.read_csv("train_dataset copy.csv")
test_dataset = pd.read_csv("test_dataset copy.csv")

# Define target variable
y_train = train_dataset["demand"]
y_test = test_dataset["demand"]

# Convert date column to datetime format
train_dataset['date'] = pd.to_datetime(train_dataset['date'])
test_dataset['date'] = pd.to_datetime(test_dataset['date'])

#Make school_day and holiday into 1 for yes and 0 for no
train_dataset['school_day'] = train_dataset['school_day'].replace({'yes': 1, 'no': 0})
test_dataset['school_day'] = test_dataset['school_day'].replace({'yes': 1, 'no': 0})

train_dataset['holiday'] = train_dataset['holiday'].replace({'yes': 1, 'no': 0})
test_dataset['holiday'] = test_dataset['holiday'].replace({'yes': 1, 'no': 0})


# Select features (excluding target variable)
x_train_name = train_dataset.columns.drop("demand")
x_train = train_dataset[x_train_name].apply(pd.to_numeric, errors='coerce').fillna(0).values
x_test_name = test_dataset.columns.drop("demand")
x_test = test_dataset[x_test_name].apply(pd.to_numeric, errors='coerce').fillna(0).values

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

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, demand_pred, alpha=0.6)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Energy Demand")
plt.show()

# print the betas
# find statistical significance
# what are the most important parameters?
# R direct functions 
# scikitlearn linear regression
# find the p-values for each parameter
# test a couple of models too
# lasso/ridge regression
# importance of variables
# understand data even more
# key part of ml: understand what part of your data is most important
# lasso/ridge: idea -> mathematical penalty applied for having too many variables

# create new linear regression for p-values
# second model 
# compare both models
# what we have taken away from the data
# table: a lot of careful analysis