# import tensorflow
# import keras
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style
data = pd.read_csv("student-mat.csv", sep=";")
print(data)

data = data[["G1", "G2", "G3", "absences"]]
print(data)
predict = "G3"
X = np.array(data.drop([predict],1))
Y = np.array(data[predict])
best = 0
## splitting the data. 10% for testing and 90% for training.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
'''Run the for loop a couple of times and then comment it out.'''
"""
for _ in range(30):

    #importing linear regression from sklearn so we can train our data model to use
    linear = linear_model.LinearRegression()
    # calling the fit function to train. The x-axis represents the categories we are trying to use to predict with y the final score

    linear.fit(x_train,y_train)
    #test the model against the test data we saved earlier.
    accuracy = linear.score(x_test, y_test)
    print(accuracy)
    if accuracy > best:
        best = accuracy
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)
"""
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print(f"coefficient : /n {linear.coef_}")
print(f"Intercept : /n {linear.intercept_}")
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(f"actual prediction: {predictions[x]}", f"input data {x_test[x]}",f"actual result: {y_test[x]}")
#don't want to retrain model every single time and save the highest scoring model.

P = "G1"
style.use("ggplot")
pyplot.scatter(data[P],data["G3"])
pyplot.xlabel(P)
pyplot.ylabel("final grade")
pyplot.show()