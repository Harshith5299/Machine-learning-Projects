#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

import scipy.io
mat = scipy.io.loadmat("C:\\Users\\user\\Desktop\\mnist_data.mat")

trX= mat['trX']
tsY=mat['tsY']
trY = mat['trY']
tsX= mat['tsX']
trX7=[] #For storing the trX values of class '7'
trX8=[] #For storing the trX values of class '8'

## Naive Bayes Algorithm Starts

for i in range(len(trX)):
    if trY[0][i]==0:
        trX7.append(trX[i][:]) # Adding the values of data pertaining to class '7' using append
    else:
        trX8.append(trX[i][:]) # Adding the values of data pertaining to class '8' using append

trX_mean =np.mean(trX, axis = 1) # Calculating the mean of all the features for a single data row in training dataset : mean(trX [1][1], trX [1][2],......trX[1][728])
trX_sd = np.std(trX, axis = 1) # Calculating the standard deviation of all the features for a single data row in training dataset : sd(trX [1][1], trX [1][2],......trX[1][728])
trX_new=[trX_mean]+[trX_sd] # Combining the resulting column matrices into a single 2-column matrix

tsX_mean =np.mean(tsX, axis = 1) # Calculating the mean of all the features for a single data row in testing dataset : mean(trX [1][1], trX [1][2],......trX[1][728])
tsX_sd = np.std(tsX, axis = 1) # Calculating the standard deviation of all the features for a single data row in training dataset : sd(trX [1][1], trX [1][2],......trX[1][728])
tsX_new=[tsX_mean]+[tsX_sd] # Combining the resulting column matrices into a single 2-column matrix

trX7_mean=[] 
trX8_mean=[]
trX7_sd=[]
trX8_sd=[]
for i in range(len(trX)):
    if trY[0][i]==0:
        trX7_mean.append(trX_mean[i]) # dataset for storing the mean values of the training data corresponding to class '7'
        trX7_sd.append(trX_sd[i]) # dataset for storing the standard deviation values of the training data corresponding to class '7'
    else:
        trX8_mean.append(trX_mean[i]) # dataset for storing the mean values of the training data corresponding to class '8'
        trX8_sd.append(trX_sd[i]) # dataset for storing the standard deviation values of the training data corresponding to class '8'


tsX7_mean=[]
tsX8_mean=[]
tsX7_sd=[]
tsX8_sd=[]
for i in range(len(tsX)):
    if tsY[0][i]==0:
        tsX7_mean.append(tsX_mean[i]) # dataset for storing the mean values of the testing data corresponding to class '7'
        tsX7_sd.append(tsX_sd[i]) # dataset for storing the standard deviation values of the testing data corresponding to class '7'
    else:
        tsX8_mean.append(tsX_mean[i]) # dataset for storing the mean values of the testing data corresponding to class '8'
        tsX8_sd.append(tsX_sd[i]) # dataset for storing the standard deviation values of the testing data corresponding to class '8'

trX7_mean_mean =np.mean(trX7_mean, axis = 0) # mean of the mean of training data of class '7' which is a single value 
trX7_mean_sd = np.mean(trX7_sd, axis = 0) # mean of the standard deviation of training data of class '7' which is a single value
trX8_mean_mean =np.mean(trX8_mean, axis = 0) # mean of the mean of training data of class '8' which is a single value
trX8_mean_sd = np.mean(trX8_sd, axis = 0) # mean of the standard deviation of training data of class '8' which is a single value

trX7_var_mean =np.var(trX7_mean, axis = 0) # variance of the mean of training data of class '7' which is a single value
trX7_var_sd = np.var(trX7_sd, axis = 0) # variance of the standard deviation of training data of class '7' which is a single value
trX8_var_mean =np.var(trX8_mean, axis = 0) # variance of the mean of training data of class '8' which is a single value
trX8_var_sd = np.var(trX8_sd, axis = 0) # variance of the standard deviation of training data of class '7' which is a single value

pY_trX7 = len(trX7)/len(trX) #prior training data probability of class '7'
temp1 = np.exp(-0.5*(((trX_mean-trX7_mean_mean)**2)/trX7_var_mean))
temp2 = temp1/math.sqrt(2*np.pi*trX7_var_mean)
temp3 = np.exp(-0.5*(((trX_sd-trX7_mean_sd)**2)/trX7_var_sd))
temp4 = temp3/math.sqrt(2*np.pi*trX7_var_sd)
pXy_trX7 = temp2*temp4 #liklihood function
pYx_trX7 = pY_trX7*pXy_trX7 #posterior training probability of class '7' without being divided by the pdf

pY_tsX7 = len(tsX7_mean)/len(tsX) #prior testing data probability of class '7'
temp5 = np.exp(-0.5*(((tsX_mean-trX7_mean_mean)**2)/trX7_var_mean))
temp6 = temp5/math.sqrt(2*np.pi*trX7_var_mean)
temp7 = np.exp(-0.5*(((tsX_sd-trX7_mean_sd)**2)/trX7_var_sd))
temp8 = temp7/math.sqrt(2*np.pi*trX7_var_sd)
pXy_tsX7 = temp6*temp8 #liklihood function
pYx_tsX7 = pY_trX7*pXy_tsX7 #posterior testing probability of class '7' without being divided by the pdf

pY_trX8 = len(trX8)/len(trX) #prior training data probability of class '8'
temp9 = np.exp(-0.5*(((trX_mean-trX8_mean_mean)**2)/trX8_var_mean))
temp10 = temp9/math.sqrt(2*np.pi*trX8_var_mean)
temp11 = np.exp(-0.5*(((trX_sd-trX8_mean_sd)**2)/trX8_var_sd))
temp12 = temp11/math.sqrt(2*np.pi*trX8_var_sd)
pXy_trX8 = temp10*temp12 #liklihood function
pYx_trX8 = pY_trX8*pXy_trX8 #posterior training probability of class '8' without being divided by the pdf

pY_tsX8 = len(tsX8_mean)/len(tsX) #prior testing data probability of class '8'
temp13 = np.exp(-0.5*(((tsX_mean-trX8_mean_mean)**2)/trX8_var_mean))
temp14 = temp13/math.sqrt(2*np.pi*trX8_var_mean)
temp15 = np.exp(-0.5*(((tsX_sd-trX8_mean_sd)**2)/trX8_var_sd))
temp16 = temp15/math.sqrt(2*np.pi*trX8_var_sd)
pXy_tsX8 = temp14*temp16 #liklihood function
pYx_tsX8 = pY_tsX8*pXy_tsX8 #posterior testing probability of class '8' without being divided by the pdf

trX_result=[]

tsX_result=[]

for j in range(2002):
    if pYx_tsX7[j]>=pYx_tsX8[j]: # comparing the posterior testing probability results and based on that, adding the predictions (i.e class '7' = 0 or class '8' = 1) to a result dataset
        tsX_result.append(0) 
        
    else:
        tsX_result.append(1)
        
m=0
for z in range(2002):
    if tsX_result[z]==tsY[0][z]: # counting the number of accurate predictions
        m=m+1

accuracy1= (m/2002)*100 

print("Accuracy of Naive Bayes: ",accuracy1)

n=0
for p in range(2002):
    if tsX_result[p]==tsY[0][p]==0: # counting the number of accurate predictions for 7
        n=n+1

accuracy7= (n/1028)*100 

print("Accuracy of predicting 7 using Naive Bayes: ",accuracy7)

o=0
for q in range(2002):
    if tsX_result[q]==tsY[0][q]==1: # counting the number of accurate predictions for 7
        o=o+1

accuracy8= (o/974)*100 

print("Accuracy of predicting 8 using Naive Bayes: ",accuracy8)


# In[ ]:


## Logistic regression


# In[7]:


trX_new=np.asarray(trX_new)
tsX_new=np.asarray(tsX_new)
from math import exp

def predict(row, w): # 'row' is a row of the matrix trX_new
    yhat = w[0] 
    for i in range(len(row)): 
        yhat = yhat + w[i+1]*row[i]
    return 1.0/(1.0+ np.exp(-yhat)) 

def coefficients(train, learn_rate, epoch):
    weights = [0,0,0] #set weights as 0 initially
    for epoch in range(epoch): 
        
        for row in train: 
            yhat=predict(row, weights)
            error=row[-1]-yhat
            weights[0]= weights[0] + learn_rate*error*yhat*(1-yhat)
            for i in range(len(row)):
                weights[i+1] = weights[i+1]+ learn_rate*error*yhat*(1-yhat)*row[i]
    return weights

epoch= 300
learn_rate=0.005

weights = coefficients(trX_new.T, learn_rate, epoch) # Gradient Descent algorithm starts
print("weights:",weights)

predicted_values=[]
values_after_regression=[]
for i in range(0,tsX_new.shape[1]):
    predicted_values.append(predict(tsX_new.T[i],weights))
    if(predicted_values[i]>0.3093): ## set the threshold value as 0.3093
        values_after_regression.append(1)
    else:
        values_after_regression.append(0)

accurate_prediction=0
for i in range(0,tsX_new.shape[1]):
    if(values_after_regression[i]==tsY[0][i]):
        accurate_prediction = accurate_prediction+1 #counting the number of accurate predictions
        
print("Accuracy of logistic Regression: ",accurate_prediction/2002)        
        
accurate_prediction_7=0
for i in range(0,tsX_new.shape[1]):
    if(values_after_regression[i]==tsY[0][i]==0):
        accurate_prediction_7= accurate_prediction_7+1 #counting the number of accurate predictions for 7

print("Accuracy of logistic Regression for predicting 7: ",accurate_prediction_7/1028)

accurate_prediction_8=0
for i in range(0, tsX_new.shape[1]):
    if(values_after_regression[i]==tsY[0][i]==1):
        accurate_prediction_8=accurate_prediction_8+1 #counting the number of accurate predictions
        
print("Accuracy of logistic Regression for predicting 8: ",accurate_prediction_8/974)


# In[ ]:




