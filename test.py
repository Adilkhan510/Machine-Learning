import tensorflow
import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np

data = pd.read_csv("student-mat.csv", sep=";")
print(data)

data = data[["G1", "G2", "G3", "absences"]]
print(data)
predict = "G3"
X = np.array(data.drop([predict],1))
Y = np.array(data[predict])
## splitting the data. 10% for testing and 90% for training.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

#importing linear regression from sklearn so we can train our data model to use
linear = linear_model.LinearRegression()
# calling the fit function to train. The x-axis represents the categories we are trying to use to predict with y the final score

linear.fit(x_train,y_train)
#test the model against the test data we saved earlier.
accuracy = linear.score(x_test, y_test)
print(accuracy)