#I will use the Iris dataset to train Linear Regression Model

#import libraries
from sklearn.linear_model import LinearRegression
from sklearn import datasets

#load dataset
[X_train, y_train] =datasets.load_iris(return_X_y=True)

model=LinearRegression()
model.fit(X_train,y_train)

#after fitting let's see what do we get as the coefficients and the intercepts
print("Model Coefficients are "+ str(model.coef_))
print("Model intercepts is " + str(model.intercept_))
