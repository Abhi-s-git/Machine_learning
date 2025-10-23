import numpy as np #importing libraries
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')#read data set 
x = dataset.iloc[:,:-1].values#slicing and storing
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split #splitting data for testing and training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression#importing linear regression model
lr = LinearRegression()
lr.fit(x_train,y_train)#fit is used to learn so here x,y are trained for linear regresion

y_predict=lr.predict(x_test)

# x_train=x_train.values
# y_train=y_train.values
# x_test=x_test.values
# y_test=y_test.values

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('salary vs experience(test set)')
plt.xlabel('years of exp')
plt.ylabel('salary')
plt.show()