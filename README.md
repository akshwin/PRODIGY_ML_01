# House Price Prediction

This project is a part of Internship in Prodigy InfoTech which aims to predict house prices based on various features such as square footage, number of bedrooms, and bathrooms. The dataset used for this project is loaded from a CSV file named 'Housing.csv', and the analysis includes exploratory data analysis (EDA), visualization, and model building.

## Exploratory Data Analysis

The exploratory data analysis (EDA) phase involves understanding the dataset's structure, checking for missing values, and exploring the distribution of features. Here's a snippet of the EDA section:

```python
# Importing libraries
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Loading the dataset
df = pd.read_csv('Housing.csv')

# Displaying basic information about the dataset
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.columns)
print(df.duplicated().sum())
```

## Visualization

Visualization is crucial for understanding the relationships between different features. This project includes pair plots and a pie chart to visualize some key features:

```python
# Pair plot
sns.pairplot(data=df)
plt.title('Pair Plot of Features')
plt.show()

# Pie chart
labels = ['bedrooms', 'bathrooms', 'stories']
sizes = [20, 45, 30]
colors = ['lightcoral', 'lightskyblue', 'lightgreen']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.title('Pie Chart ')
plt.show()
```

## Model Building

The main goal is to predict house prices using machine learning models. Two regression models are implemented: Linear Regression and Decision Tree Regressor. The dataset is split into training and testing sets, and each model is trained and evaluated.

```python
# Splitting the dataset
x = df.iloc[:, 0:4]
y = df['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Linear Regression
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)
lr_model_test_score = lr_model.score(x_test, y_test)
print(f"Linear Regression Test Score: {lr_model_test_score}")

# Decision Tree Regressor
dTree = DecisionTreeRegressor(criterion='friedman_mse', random_state=1)
dTree.fit(x_train, y_train)
dTree_test_score = dTree.score(x_test, y_test)
print(f"Decision Tree Regressor Test Score: {dTree_test_score}")
```

## Model Comparison

Several regression models are implemented and compared for accuracy. The following models are included: Decision Tree Regressor, Bagging Regressor, Random Forest Regressor, and XGB Regressor.

```python
# Bagging Regressor
from sklearn.ensemble import BaggingRegressor
bgcl = BaggingRegressor(estimator=dTree, n_estimators=50, random_state=1)
bgcl.fit(x_train, y_train)
bgcl_test_score = bgcl.score(x_test, y_test)
print(f"Bagging Regressor Test Score: {bgcl_test_score}")

# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
rfcl = RandomForestRegressor(n_estimators=50, random_state=1, max_features=12)
rfcl.fit(x_train, y_train)
rfcl_test_score = rfcl.score(x_test, y_test)
print(f"Random Forest Regressor Test Score: {rfcl_test_score}")

# XGB Regressor
from sklearn.ensemble import GradientBoostingRegressor
gbcl = GradientBoostingRegressor(n_estimators=50, random_state=1)
gbcl.fit(x_train, y_train)
gbcl_test_score = gbcl.score(x_test, y_test)
print(f"Gradient Boosting Regressor Test Score: {gbcl_test_score}")
```

## Conclusion

In conclusion, this project provides a step-by-step guide on predicting house prices using machine learning models. The accuracy of each model is compared, and the Gradient Linear Regressor demonstrates the highest accuracy in this specific scenario. 

 **Linear Regression: 100.0%**
  - The Linear Regression model achieved perfect accuracy on the test dataset.

- **Decision Tree Regressor: 99.2%**
  - The Decision Tree Regressor demonstrated a high accuracy of 99.2% on the test dataset.

- **Bagging Regressor: 99.5%**
  - The Bagging Regressor, employing an ensemble of Decision Trees, achieved an accuracy of 99.5%.

- **Random Forest Regressor: 99.5%**
  - The Random Forest Regressor, a powerful ensemble model, exhibited an accuracy of 99.5% on the test dataset.

- **XGB Regressor: 99.6%**
  - The XGB Regressor, utilizing gradient boosting, outperformed other models with an accuracy of 99.6%.
