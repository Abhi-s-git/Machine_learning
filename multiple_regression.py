import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
df = pd.read_csv('price.csv')

# Step 2: Clean 'price' column (target variable)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Step 3: Drop irrelevant columns (non-numeric and non-useful)
df.drop(columns=['ID', 'name'], inplace=True)

# Step 4: Drop rows with any missing values
df = df.dropna()

# Step 5: Remove outliers from price (top 1%)
df = df[df['price'] < df['price'].quantile(0.99)]

# Step 6: Separate features (X) and target (y)
x = df.drop(columns=['price'])
y = df['price']

# Step 7: Convert categorical variables to numeric (one-hot encoding)
x = pd.get_dummies(x, drop_first=True)

# Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Step 9: Train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Step 10: Train the Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Step 11: Predict
y_pred = lr.predict(x_test)

# Step 12: Evaluation
from sklearn.metrics import r2_score, mean_squared_error
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
