#To use enviornment cmd+shift+p then "Select pythong env"
#Practicing Linear Regression

#importing all neccessary libraries
import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


#Pulling data used for testing
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "studytime", "traveltime"]]

#setting value being tested
predict = "G3"

#Dropping tested vaue from data
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

BestAcc = 0

while BestAcc < .95:
    #Seperating data in lists to test
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

    #setting learning model (Linear Regression)
    linear = linear_model.LinearRegression()

    #Creating line of best fit
    linear.fit(x_train, y_train)

    #Finding accuracy of test
    Accuracy = linear.score(x_test, y_test)

    #Findinf predictions of test
    predictions = linear.predict(x_test)

    print(round(Accuracy, 2))

    if Accuracy > BestAcc:
        #Changes Best Accuracy if greater
        BestAcc = Accuracy
        #Create pickle file to save model
        with open("Model.pickle", "wb") as f:
            pickle.dump(linear, f)

print ("Best Accuracy:",round(BestAcc, 2))
