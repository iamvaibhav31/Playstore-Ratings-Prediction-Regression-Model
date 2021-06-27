import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *
from sklearn.linear_model  import LinearRegression , Lasso , Ridge 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Playstore Ratings Prediction Regression Model\\googleplaystore.csv")


def sizedatapreproses(size):
    size = size.lower()
    if "m" in size:
        temp=size.replace("m","")
        if "." in temp:
            return float(temp)*1000000
        else:
            return int(temp)*1000000
    elif "k"  in size:
        temp=size.replace("k","")
        if "." in temp:
            return float(temp)*1000
        else:
            return int(temp)*1000
        
def price(price):
    return float(price*74)
    
"""print(data.isna().sum())
print(data.describe())
print(data["Rating"].value_counts())"""

data = data.dropna()

"""print(data.isna().sum())"""

data["Installs"] = data["Installs"].str.replace("+","" ,regex=True)
data["Installs"] = data["Installs"].str.replace(",","",regex=True)
data["Installs"] = data["Installs"].astype(int)
data["Size"] = data["Size"].str.replace("Varies with device","0k",regex=True)
data["Size In Bytes"]  =  data["Size"].apply(lambda x : sizedatapreproses(x))
data  = data.drop(columns=["Type","Size","Content Rating"])
data["Size In Bytes"] = data["Size In Bytes"].astype(float)
data["Price"] = data["Price"].str.replace("$","",regex=True)
data["Price"] = data["Price"].astype(float)
data["Price"] = data["Price"].apply(price)
data["Reviews"] = data["Reviews"].astype(int)
data = data[data["Installs"]>=50000]
data = data[data["Installs"]<=100000000]
data = data[data["Reviews"]>200]
data = data[data["Reviews"] <= 2000000]
data = data.reset_index(drop=True)
"""#data.to_csv("datapreprocessing_googleplaystore.csv")
print(data.head(50))
print(data.describe())
print(data.shape)
"""
x = data["Reviews"].values.reshape(-1,1)
y = data["Rating"].values.reshape(-1,1)

 


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


scaler =  StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)

# LINAR REGRESSION 
print("linear regresion ")
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_train_predict = regressor.predict(x_train)
plt.scatter(x_train,y_train ,  )
plt.plot(x_train,y_train_predict , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Linear Regression Train Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted training set :- {mean_absolute_error(x_train,y_train_predict)}")
print(f"Mean Squared error of predicted training set :- {mean_squared_error(x_train,y_train_predict)}")

y_test_predict = regressor.predict(x_test)
plt.scatter(x_test,y_test )
plt.plot(x_test,y_test_predict , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Linear Regression Test Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted test set :- {mean_absolute_error(x_test,y_test_predict)}")
print(f"Mean Squared error of predicted test set :- {mean_squared_error(x_test,y_test_predict)}")

# END
print("\n")
# LASSO
print("lasso  regression")
regressor = Lasso()
regressor.fit(x_train,y_train)
y_predict = regressor.predict(y_test)

y_train_predict = regressor.predict(x_train)
plt.scatter(x_train,y_train )
plt.plot(x_train,y_train_predict  , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Lasso Regression Train Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted training set :- {mean_absolute_error(x_train,y_train_predict)}")
print(f"Mean Squared error of predicted training set :- {mean_squared_error(x_train,y_train_predict)}")

y_test_predict = regressor.predict(x_test)
plt.scatter(x_test,y_test )
plt.plot(x_test,y_test_predict  , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Lasso Regression Test Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted test set :- {mean_absolute_error(x_test,y_test_predict)}")
print(f"Mean Squared error of predicted test set :- {mean_squared_error(x_test,y_test_predict)}")

# END
print("\n")
# RIDGE
print(" Ridge Regression ")
regressor = Ridge()
regressor.fit(x_train,y_train)
y_predict = regressor.predict(y_test)

y_train_predict = regressor.predict(x_train)
plt.scatter(x_train,y_train )
plt.plot(x_train,y_train_predict  , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Ridge Regression Train Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted traning set :- {mean_absolute_error(x_train,y_train_predict)}")
print(f"Mean Squared error of predicted training set :- {mean_squared_error(x_train,y_train_predict)}")

y_test_predict = regressor.predict(x_test)
plt.scatter(x_test,y_test )
plt.plot(x_test,y_test_predict , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Ridge Regression Test Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted test set :- {mean_absolute_error(x_test,y_test_predict)}")
print(f"Mean Squared error of predicted test set :- {mean_squared_error(x_test,y_test_predict)}")

# END
print("\n")
# DECISION TREE REGRESSOR
print("Decision tree regre")

regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)
y_predict = regressor.predict(y_test)

y_train_predict = regressor.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,y_train_predict  , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Decision Tree Regression Train Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted traning set :- {mean_absolute_error(x_train,y_train_predict)}")
print(f"Mean Squared error of predicted training set :- {mean_squared_error(x_train,y_train_predict)}")

y_test_predict = regressor.predict(x_test)
plt.scatter(x_test,y_test )
plt.plot(x_test,y_test_predict , linewidth=0.9)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Decision Tree Regression Test Set Prediction Graph")
plt.show()
print(f"Mean Absolute Error of Predicted test set :- {mean_absolute_error(x_test,y_test_predict)}")
print(f"Mean Squared error of predicted test set :- {mean_squared_error(x_test,y_test_predict)}")

# END
print("\n")
# RANDOM FOREST REGRESSOR
print("Random Forest Regression ")

regressor = RandomForestRegressor(n_estimators=20)
regressor.fit(x_train,y_train)
y_predict = regressor.predict(y_test)

y_train_predict = regressor.predict(x_train)
plt.scatter(x_train,y_train)
plt.plot(x_train,y_train_predict , linewidth=0.5)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Random Tree Regression Train Set Prediction Graph")
plt.show()

print(f"Mean Absolute Error of Predicted traning set :- {mean_absolute_error(x_train,y_train_predict)}")
print(f"Mean Squared error of predicted training set :- {mean_squared_error(x_train,y_train_predict)}")

y_test_predict = regressor.predict(x_test)
plt.scatter(x_test,y_test )
plt.plot(x_test,y_test_predict  , linewidth=0.5)
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.title("Random Tree Regression Test Set Prediction Graph")
plt.show()


print(f"Mean Absolute Error of Predicted test set :- {mean_absolute_error(x_test,y_test_predict)}")
print(f"Mean Squared error of predicted test set :- {mean_squared_error(x_test,y_test_predict)}")

# END
print("\n")
