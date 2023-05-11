#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:05:55 2022

@author: apple1
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data=pd.read_excel("iris.xls")

y=data['Classification']
x=data.drop(['Classification'],axis=1)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# training the data
from sklearn.ensemble import RandomForestClassifier
Randomforest=RandomForestClassifier()
Randomforest.fit(x_train,y_train)

#fitting the model
pred=Randomforest.predict(x_test)
#Saving the model to disk
pickle.dump(Randomforest,open('model.pkl','wb'))
