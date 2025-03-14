#Support Vector Machine on IRIS dataset

#import libraries
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load dataset
iris =datasets.load_iris()

#Split the data into training and test(90-10 ratio)
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.10)

#Build Model
model=SVC(kernel='linear')

#train model
model.fit(X_train,y_train)

#predict values for test data
y_predicted=model.predict(X_test)

#print model metrics
print(classification_report(y_test,y_predicted))
