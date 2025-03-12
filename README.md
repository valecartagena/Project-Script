This is our project code for CHEMENG 177!

Project Overview:
This project aims to predict electricity demand using lasso regression and a null hypothesis test with the ultimate goal of moving toward clean energy. 

Instructions to run the code:

We use scikitlearn, numpy, and pandas as the basis for our project. Please download the following:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score


In order to use the code, do the following:
1. Data and Files preparation:
    Install in a local computer the following files:
    - lasso.py
    - last_lr.py
    - null_hypothesis.py
    Also install the dataset we are going to use:
    - dataset.csv
     Note: the dataset was obtained from: https://www.kaggle.com/datasets/aramacus/electricity-demand-in-victoria-australia?resource=download (All our scripts are linked to the dataset file name "dataset.csv)

For running the scripts (Visual Studio Code is recommended):

2. 
    Run null_hypothesis.py first to analyze the significance of each of our parameters. For this part of our code we used bootstrapping with 95% of Confidence Interval. Use all of the parameters. Demand will be the dependent variable.

3. 
    Run last_lr.py, which will show the linear regression with only the statistically significant parameters as determined by the null hypothesis. Demand will be the dependent variable. Use only the statistically significant parameters as the independent.

4. 
    Run lasso.py, which uses a different method of determining the statistically significant parameters. Pass in all of the parameters except for demand as the independent, and pass in demand as the dependent. 


