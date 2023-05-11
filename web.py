#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 20:05:17 2022

@author: apple1
"""

from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
   SL= float(request.form['SL'])
   SW= float(request.form['SW'])
   PL= float(request.form['PL'])
   PW= float(request.form['PW'])
   arr = np.array([[SL, SW, PL, PW]])
   pred = model.predict(arr)
   return render_template ('result.html',prediction_text="The iris species is {}".format(pred))
if __name__=='__main__':
    app.run(port=8000)
