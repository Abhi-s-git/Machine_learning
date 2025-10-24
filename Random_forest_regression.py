from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
boston = load_boston()
X, y = boston.data, boston.target
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor()
rf_reg.fit(X_train, y_train)
#evaluation
y_pred = rf_reg.predict(X_test)
print("Random Forest Regression MSE:", mean_squared_error(y_test, y_pred))
