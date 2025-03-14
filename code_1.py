#SUM OF TWO NUMBERS

#import libraries

from sklearn import linear_model
import numpy as np


#create training data
input_data = np.random.randint(50,size=(20,2))
input_sum = np.zeros(len(input_data))

for row in range(len(input_data)):
  input_sum[row]=input_data[row][0] +input_data[row][1]

#Build the model
linear_regression_model = linear_model.LinearRegression(fit_intercept=False)

#train the model
linear_regression_model.fit(input_data,input_sum)

#predict for the new data
predicted_sum=linear_regression_model.predict([[60,24]])

#print the result
print(str(predicted_sum))
print(str(linear_regression_model.coef_))

     
