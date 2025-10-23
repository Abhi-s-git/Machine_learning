import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('price.csv')
df['price']=pd.to_numeric(df['price'],errors='coerce')

df.drop(columns=['ID', 'name'], inplace=True)
df = df.dropna()

df = df[df['price'] < df['price'].quantile(0.99)]

x = df.drop(columns=['price'])
y = df['price']

df = pd.getdummies(x,drop_first = True )

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)