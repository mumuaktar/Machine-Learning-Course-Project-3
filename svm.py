import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, svm, metrics
import time
import datetime as dt

x_train = pd.read_pickle('./datasets/train_processed1.pkl')

Y_train = pd.read_csv('./datasets/train_labels.csv')



x_train = np.asarray(x_train)
print(x_train.shape)

x_test = np.load('./datasets/test_processed.npy')
x_test= np.asarray(x_test)
print(x_test.shape)


y_train = []
for i in range(x_train.shape[0]):
    y_train.append(Y_train.iloc[i]['Category'])
    
y_train = np.asarray(y_train)
print(y_train.shape)

text_train, text_test, y_train1, y_test = train_test_split(
    x_train, y_train, test_size=0.0625, random_state=42)

x_train1 = text_train.reshape(text_train.shape[0], 64*64)
text_test = text_test.reshape(text_test.shape[0], 64*64)

steps = [ ('SVM', SVC(kernel='poly',gamma=10,  C=0.001))]
pipeline = Pipeline(steps)

start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
pipeline.fit(x_train1, y_train1)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


y_pred = pipeline.predict(text_test)
print(accuracy_score(y_test, y_pred))
      
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n%s" % cm)



#print "confusion matrix: \n ", confusion_matrix(y_test, y_pred)