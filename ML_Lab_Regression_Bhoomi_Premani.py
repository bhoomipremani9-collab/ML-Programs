# ML Lab - Regression
# Extracted from the provided Jupyter Notebook
# Original notebook details:
# Name: Bhoomi Premani
# Roll No: 2
# Section: C2-B1
#
# Dataset required: USA_Housing.csv


# ===== Notebook Code Cell 1 =====
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ===== Notebook Code Cell 2 =====
df = pd.read_csv("USA_Housing.csv")
df.head()


# ===== Notebook Code Cell 3 =====
df.tail()


# ===== Notebook Code Cell 4 =====
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(df['Price'], bins=30)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# ===== Notebook Code Cell 5 =====
df['Price'].hist(bins=30, figsize=(8,5))
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# ===== Notebook Code Cell 6 =====
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm')
plt.show()


# ===== Notebook Code Cell 7 =====
df.hist(figsize=(12,10), bins=30)
plt.show()


# ===== Notebook Code Cell 8 =====
X=df[['Avg. Area Income']]
y=df['Price']


# ===== Notebook Code Cell 9 =====
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ===== Notebook Code Cell 10 =====
from sklearn.linear_model import LinearRegression
slr_model=LinearRegression()
slr_model.fit(X_train, y_train)


# ===== Notebook Code Cell 11 =====
y_pred=slr_model.predict(X_test)


# ===== Notebook Code Cell 12 =====
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

print("-------------------------")

print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score :", r2_score(y_test, y_pred))


# ===== Notebook Code Cell 13 =====
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color='skyblue', label='Actual Data')

plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')

plt.xlabel("Average Area Income")
plt.ylabel("House Price")
plt.title("Linear Regression: Actual vs Predicted")

plt.legend()

plt.grid(True)

plt.show()


# ===== Notebook Code Cell 14 =====
from sklearn.model_selection import GridSearchCV


# ===== Notebook Code Cell 15 =====
from sklearn.linear_model import Ridge
ridge= Ridge()
ridge.fit(X_train, y_train)
ridge_pred=ridge.predict(X_test)
print("Default Ridge R^2:",r2_score(y_test, ridge_pred))


# ===== Notebook Code Cell 16 =====
param_grid={'alpha':[0.001,0.01,0.1,1,10,100]}


# ===== Notebook Code Cell 17 =====
grid_ridge=GridSearchCV(estimator=Ridge(),param_grid=param_grid,scoring='r2',cv=4)


# ===== Notebook Code Cell 18 =====
grid_ridge.fit(X_train, y_train)


# ===== Notebook Code Cell 19 =====
print(grid_ridge.best_score_)


# ===== Notebook Code Cell 20 =====
best_ridge=grid_ridge.best_estimator_
ridge_pred=best_ridge.predict(X_test)


# ===== Notebook Code Cell 21 =====
print("MAE:", mean_absolute_error(y_test,ridge_pred))
print("MsE:", mean_squared_error(y_test,ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,ridge_pred)))
print("R2:", r2_score(y_test,ridge_pred))


# ===== Notebook Code Cell 22 =====
from sklearn.linear_model import Lasso
param_grid = {'alpha':[0.001,0.01,0.1,1,10]}


# ===== Notebook Code Cell 23 =====
grid_lasso=GridSearchCV(Lasso(max_iter=5000),param_grid,cv=5,scoring='r2')
grid_lasso.fit(X_train,y_train)


# ===== Notebook Code Cell 24 =====
print(grid_lasso.best_params_)


# ===== Notebook Code Cell 25 =====
lasso_pred = grid_lasso.predict(X_test)
print(r2_score(y_test,lasso_pred))


# ===== Notebook Code Cell 26 =====
print("MAE:", mean_absolute_error(y_test,ridge_pred))
print("MsE:", mean_squared_error(y_test,ridge_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,ridge_pred)))
print("R2:", r2_score(y_test,ridge_pred))


# ===== Notebook Code Cell 27 =====
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


mlr = LinearRegression()


mlr.fit(X_train, y_train)


y_pred = mlr.predict(X_test)


print("Default Linear Regression R^2:", r2_score(y_test, y_pred))


print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# ===== Notebook Code Cell 28 =====
predictions = mlr.predict(X_test)


# ===== Notebook Code Cell 29 =====
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions, color='yellow',label='Predictions')

plt.plot([min(y_test), max(y_test)],[min(y_test), max(y_test)], color='red',linewidth=2, label='Perfect Fit')

plt.title('Actual vs predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()


# ===== Notebook Code Cell 30 =====
income=float(input("Enter Average Area Income: "))
house_age=float(input("Enter Average Area House Age: "))
rooms=float(input("enter average area number of rooms :"))
bedrooms=float(input("enter average area number of bedrooms:"))
population=float(input("enter area population: "))

test_input=np.array([[income, house_age, rooms, bedrooms, population]])

predicted_price=mlr.predict(test_input)

print("\n Predicted house price: ${:,.2f}".format(predicted_price[0]))
