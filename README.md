# House-Price-Prediction
#this is machine learning project 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Read Data
house = pd.read_csv('Boston (1).csv')
house.head()

# Shape of Data
house.shape

# Data information
house.info()

# Checking Null Values
house.isna().sum()

# Checking Duplicate Values
house.duplicated().sum()

# Summary of data
house.describe()

house.columns


# Histogram of Housing Prices (MEDV)

sns.distplot(house['MEDV'], bins=20, kde=True)
plt.title('Distribution of Housing Prices (MEDV)')
plt.xlabel('Median Housing Price ($1000s)')
plt.ylabel('Frequency')
plt.show()



# List of features
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT'] 

# Create a scatter plot and boxplot for each feature side by side
for feature in features:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))  # Create a new figure with 1 row and 2 columns

    # Scatter plot of feature with target variable
    axes[0].scatter(house[feature], house['MEDV'])
    axes[0].set_title(f'Scatter plot of {feature} with House Price')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('House Price')

    # Boxplot of the feature
    axes[1].boxplot(house[feature])
    axes[1].set_title(f'Boxplot for {feature}')
    axes[1].set_xlabel('Feature')
    axes[1].set_ylabel('Values')

    plt.tight_layout()  
    plt.show();

rad_medv_mean = house.groupby('RAD')['MEDV'].mean().reset_index()
rad_medv_mean


# Bar Plot Average House Price By Accessibility of Road Highways

sns.barplot(x='RAD', y='MEDV', data=rad_medv_mean, color='orange',edgecolor='black')
plt.title('Average House Price By Accessibility of Road Highways')
plt.xlabel('Accessibility of Road Highways')
plt.ylabel('Mean Housing Price ($1000s)')
plt.show();


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(house.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Boston Housing Features')
plt.show()


# Split the Data 
X = house.drop(columns=['MEDV']) #features
y = house['MEDV'] #target variable


# Splitting Data for Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=2)


# shape of spiltted data
print("The shape of X_train :",X_train.shape)
print("The shape ofX_test :",X_test.shape)
print("The shape of y_train :",y_train.shape)
print("The shape of y_test :",y_test.shape)


# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor 
dtr = DecisionTreeRegressor(max_depth=5)


# Predictions of  decision Tree Regressor on Testing Data
y_pred_dtr=dtr.predict(X_test)


# Random Forest Regressor 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(max_depth = 10, min_samples_leaf = 1, min_samples_split = 2, n_estimators = 200)


# Fit the model on Training datset
rfr.fit(X_train,y_train)


# Predictions of  Ranforest Forest Regressor on Testing Data
y_pred_rfr = rfr.predict(X_test)


# Accuracy Score of Model
from sklearn.metrics import mean_absolute_percentage_error # Import the function

error = mean_absolute_percentage_error(y_pred_rfr,y_test)
print("Accuracy of Random Forest Regressor is :%.2f "%((1 - error)*100),'%')

