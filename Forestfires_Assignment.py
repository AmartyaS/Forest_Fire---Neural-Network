# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 09:15:19 2021

@author: ASUS
"""

#Importing all the necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

#Loading the dataset
file=pd.read_csv(r"D:\Data Science Assignments\Python-Assignment\Neural Network\forestfires.csv")

#Exploring the dataset
file.columns
file.dtypes
file.size_category.value_counts()

#Splitting the dataset into training and testing data
train=file.iloc[:,2:30]
test=file.loc[:,'size_category']
x_train,x_test,y_train,y_test=train_test_split(train, test, test_size=0.2)

#Normalizing the dataset
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#Model1 with Rectified Linear Activation function, No. of neurons and hidden layers =30
model1=MLPClassifier(hidden_layer_sizes=(30,30),random_state=4)
model1.fit(x_train,y_train)
#Prediction of values based on the model formed
pred1_train=model1.predict(x_train)
pred1_test=model1.predict(x_test)
#Checking the model accuracy
pd.crosstab(y_train,pred1_train) 
pd.crosstab(y_test,pred2_test)
np.mean(y_train==pred1_train) #Training Accuracy
np.mean(y_test==pred1_test)   #Testing Accuracy


#Model2 with Hyperbolic Tangent Activation function, No. of neurons =40 and hidden layers =30
model2=MLPClassifier(activation='tanh',hidden_layer_sizes=(40,30),random_state=4)
model2.fit(x_train,y_train)
#Prediction of values based on the model formed
pred2_train=model2.predict(x_train)
pred2_test=model2.predict(x_test)
#Checking the model accuracy
pd.crosstab(y_train,pred2_train)
pd.crosstab(y_test,pred2_test)
np.mean(y_train==pred2_train)  #Training Accuracy
np.mean(y_test==pred2_test)    #Testing Accuracy

