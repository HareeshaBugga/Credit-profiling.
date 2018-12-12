import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
import os
import time
import sys

t=time.time()

#read data from csv
readCSV=pd.read_csv('german_credit.csv')

X=readCSV.drop('Creditability',axis=1)
Y=readCSV['Creditability']


#to fix Scikit NaN or infinity error message
X = Imputer().fit_transform(X)

#split dataset into training and testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

#Logistic Regression
print "\nLogistic Regression"
md=LogisticRegression()
fitted_model=md.fit(X_train,Y_train)

predictions=fitted_model.predict(X_test)
print "Prediction of class of test data"
print predictions

print "Confusion matrix"
print confusion_matrix(Y_test,predictions)
print "Accuracy score"
print accuracy_score(Y_test,predictions)

#Random Forests classifier
print "\nRandom Forest"
rf=RandomForestClassifier(n_estimators=1000)
rf.fit(X_train,Y_train)

#classification is done here
prediction=rf.predict(X_test)

print "Prediction of class of test data"
print prediction

print "Confusion matrix"
print confusion_matrix(Y_test,prediction)

print "Accuracy score"
print accuracy_score(Y_test,prediction)

#Naive Bayes' Classifier
print "\nNaive Bayes Classifier"
gnb = GaussianNB()
gnb.fit(X_train,Y_train)

#classification is done here
prediction=gnb.predict(X_test)

print "Prediction of class of test data"
print prediction

print "Confusion matrix"
print confusion_matrix(Y_test,prediction)

print "Accuracy score"
print accuracy_score(Y_test,prediction)


print "Total time taken:\t"
print time.time()-t

