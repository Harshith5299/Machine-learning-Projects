#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Project by Harshith Chittajallu, ASU ID: 1218707243 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

import scipy.io
mat = scipy.io.loadmat("C:\\Users\\user\\Desktop\\AllSamples.mat")
data = mat['AllSamples']            # Stored the data from the file in 'data' array
datanew= np.array(data) #converting the stored array into an np.ndarray
data_x =[] # For storing x coordinates of the dataset
data_y =[] # For storing y coordinates of the dataset
for i in range(len(data)): # loop which stores the corresponding coordinates in the arrays
    data_x.append(data[i][0])
    data_y.append(data[i][1])
data_xnew = np.array(data_x) #converting the list of x coordinates into a numpy array
data_ynew= np.array(data_y) # converting the list of y coordinates into a numpy array
data_xnew = data_xnew.reshape((300,1)) # reshaping the numpy array into a 2-D matrix with 300 rows
data_ynew = data_ynew.reshape((300,1)) # same as above
plt.scatter(data_xnew,data_ynew, s=10) # for plotting the unclassified data
plt.title("unclassified data")
plt.xlabel("x-label")
plt.ylabel("y-label")
plt.show()


# In[3]:


# STRATEGY 1 K-MEANS
s=int(input('Enter the number of k clusters from 2 to 10 ')) # for taking the input value of the number of clusters required for classification 
if 2 <= s <= 10: # only accepts cluster input values from 2 to 10
    k = datanew[np.random.choice(datanew.shape[0], size=s, replace=False), :] # selects 's' random rows from the original data set and stores them as an 's'X2 matrix. It represents an array storing the number of initial cluster centers as desired by the input. 
    kupdate=k # variable that sets the initial centres of the algorithm
    c=0 # for counting the number of cycles until convergence
    # Algorithm begins
    for cycles in range(100): # 100 is the maximum number of cycles currently allowed for the algorithm
        dist=[[]]*300 # creates an empty distance matrix with dimensions 300x1
        dist=np.array(dist) # converts the distance matrix into a numpy array
        for f in range(s): # This loop goes on upto the input number of clusters  
            dist1=[] # List for storing all point's distances from the 'f'th cluster
        
            for p in range(len(data)): #runs as long as the lenght of the given data, which is 300
                ed =  math.sqrt((kupdate[f][0]-datanew[p][0])**2 + (kupdate[f][1]-datanew[p][1])**2) # This calculates the euclidean distance from a particular data point to the 'f'th cluster center.
                dist1.append(ed) # for adding the distances to the list
            dist1=np.array(dist1) # converting the list into a numpy array 
            dist1=dist1.reshape(300,1) # converting the above numpy array into a matrix with 300 rows and 1 column
            dist=np.append(dist,dist1,axis=1) # This was initially the empty matrix with dimensions 300x1. Now, the array which was used to store all the distances from the points to the 'f'th cluster center in above line is added to it, but it still has dimensions 300x1 as it was initially empty. After the end of the loop, it's dimensions will be 300x's'. 


        min1 =np.argmin(dist,axis=1) # This list stores the index value of the minimum value of all elements in a particular row. This list has 300 elements. For example, if the index value = 3 for the 1st element of this list, it means that the distance from cluster k3 to point p1 of the data set is minimal.   
        matf=[] # 3-D matrix for storing the list of all the 2-D data points belonging to a particular cluster in the 3rd dimension. 
        for q in range(s): # loop runs as many times as the number of clusters    
            mat1=[] # list for storing all the data points of a  particular cluster
            for i in range(len(data)): # loop runs as many times as the total data points
                if q ==min1[i]: # if the current 'q'th cluster matches any element of the list min1, the correspoding 'i'th data point is added to the list mat1 
                    mat1.append(data[i])
            matf.append(mat1) #the list containing all the points closer to the 'q'th cluster is added to the 'q'th column of the marix matf, with total dimensions as q X 300 X 2.
        knew=[] # list which will contain the updated cluster centers
        for dimen in range(s): # loop for updating the cluster centers
            x=0 # setting the initial x and y coordinates to 0
            y=0 
            if len(matf[dimen])==0: # if a situation occurs where there are no points close to a cluster center, we do not update the cluster for that particular cycle
                t1=kupdate[dimen]
                knew.append(t1) # add the old cluster center to the list holding the updated cluster centers
            else:    
                for cent in range(len(matf[dimen])): # loop to find centroid
                    x= x+matf[dimen][cent][0] 
                    y=y+matf[dimen][cent][1]
                x1=x/len(matf[dimen])
                y1=y/len(matf[dimen])
                t1=[x1,y1]
                knew.append(t1) # add the updated cluster center to the list holding the updated cluster centers
        if c==0: # if this is the first cycle, this forces the algorithm to continue
            kupdate=knew # update the list of cluster centers with the new one
            c=c+1
            continue
        elif np.array_equal(knew,kupdate)==True: # if the previous cluster centers are exactly equal to the updated cluster centers, this forces the algorithm to converge 
            break
        else:
            kupdate=knew
            c=c+1
    # kmeans algorithm ends
    c1=0
    for v1 in range(s): # algorithm to calculate cost function
        for v2 in range(len(matf[v1])):
            c1= c1 + (matf[v1][v2][0]-knew[v1][0])**2 + (matf[v1][v2][1]-knew[v1][1])**2
    fig=plt.figure(figsize=(20,10))
    color=['red','skyblue','blue','magenta','pink','orange','green','brown','yellow','black'] # array assigning colors to clusters based on index value
    leg=['k1 cluster','k2 cluster','k3 cluster','k4 cluster','k5 cluster','k6 cluster','k7 cluster','k8 cluster','k9 cluster','k10 cluster'] # array for legends
    leg2=['k1 center','k2 center','k3 center','k4 center','k5 center','k6 center','k7 center','k8 center','k9 center','k10 center']
    plot=plt.subplot(111)
    for w in range(s): # algorithm for plotting the points 's'th cluster wise 
        arr=np.array(matf[w])
        plot.scatter(arr[:,0],arr[:,1],color=color[w],label=leg[w],s=5)
        plot.scatter(knew[w][0],knew[w][1],color=color[w],label=leg2[w],marker='x',s=200) # shows the final cluster centers 
    chartBox = plot.get_position()  # these 3 lines of code are for setting the legend outside the graph
    plot.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    plot.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), shadow=True, ncol=1)
    plt.title("clustered data with 'k' random clusters")
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.show()
    print('cost function of these clusters =',c1)
    print('number of cycles required for convergence =',c)
else:
    print('Enter a number between 2 and 10')


# In[7]:


# STRATEGY 1 COST FUNCTION GRAPH
# Here, we are only calculating the corresponding cost function for 2 to 10 clusters. Hence s is taken to be 10.  
# The code is the same as above, except that it is made to run 9 times to calculate the cost function in each case of clusters from 2 to 10 
s=10
cfunc=[]
k = datanew[np.random.choice(datanew.shape[0], size=s, replace=False), :]
for z in range(2,s+1): # code made to run a total of 9 times  
    
    kupdate=k
    c=0
    

    for cycles in range(30):
        dist=[[]]*300
        dist=np.array(dist)
        for f in range(z):
            dist1=[]
        
            for p in range(len(data)):
                ed =  math.sqrt((kupdate[f][0]-datanew[p][0])**2 + (kupdate[f][1]-datanew[p][1])**2)
                dist1.append(ed)
            dist1=np.array(dist1)
            dist1=dist1.reshape(300,1)
            dist=np.append(dist,dist1,axis=1)


        min1 =np.argmin(dist,axis=1)
        matf=[]
        for q in range(z):    
            mat1=[]
            for i in range(len(data)):
                if q ==min1[i]:
                    mat1.append(data[i])
            matf.append(mat1)
        knew=[]
        for dimen in range(z):
            x=0
            y=0
            for cent in range(len(matf[dimen])):
                    x= x+matf[dimen][cent][0]
                    y=y+matf[dimen][cent][1]
            x=x/len(matf[dimen])
            y=y/len(matf[dimen])
            t1=[x,y]
            knew.append(t1)
        kupdate=knew
    c1=0
    for v1 in range(z):
        for v2 in range(len(matf[v1])):
            c1= c1 + (matf[v1][v2][0]-knew[v1][0])**2 + (matf[v1][v2][1]-knew[v1][1])**2
    cfunc.append(c1) # Stores all the cost function values from 2 to 10, which are a total of 9
clusters=[] # an array which is made to contain numbers starting from 2 to 10 in the loop below for the elbow graph
for t in range(2,s+1):
    clusters.append(t)
plt.plot(clusters, cfunc) # for plotting the elbow graph
plt.title("Elbow curve of 'k' initially random centered clusters")
plt.xlabel("number of 'k' clusters")
plt.ylabel("Cost Function")
print('cost function values are',cfunc)


# In[5]:


# STRATEGY 2 K++ MEANS
s=int(input('Enter the number of k clusters from 2 to 10 '))
if 2 <= s <= 10:
    k = datanew[np.random.choice(datanew.shape[0], size=1, replace=False), :] # We are generating only 1 point from the data set
    dist=[[]]*300
    dist=np.array(dist)

    kj=np.array(k) # Matrix which stores the generated cluster centers and initially has the first random cluster center
    for size in range(s-1): # Algorithm for generating the remaining s-1 clusters
        dist1=[]
        for p in range(len(data)):
            ed =  math.sqrt((kj[size][0]-datanew[p][0])**2 + (kj[size][1]-datanew[p][1])**2)
            dist1.append(ed)
        dist1=np.array(dist1)
        dist1=dist1.reshape(300,1)
        dist=np.append(dist,dist1,axis=1) # Upto here a distance matrix is generated which stores distances from all points to a particular cluster center in a column
        if size==0:
            k2=np.array(datanew[np.argmax(dist,axis=0)]) # If it is the 2nd cluster being calculated, the dist matrix has only 1 column and thus the data point which is the farthest from the initial center, is found using this statement which calculates the index value of the maximum distance in the matrix of 300x1 distances
        else:
            smat=np.array(np.sum(dist,axis=1)) # In all other cases, we have another matrix which stores the sum of all distances of each point to a particular cluster center and divides it by the number of cluster centers, essentially generating a matrix having the average distance of a point to all centers.
            smat2=smat/len(kj)
            smat2=smat2.reshape(300,1)
            k2=np.array(datanew[np.argmax(smat2,axis=0)]) # This finds out the point from the dataset which has the maximum average distance, found from the above matrix
        kj=np.append(kj,k2,axis=0) # Adding the new cluster center to the list of initial cluster centers
    
    kupdate=kj # setting up the initial 's' clusters. The rest of the kmeans algorithm code is the same.
    c=0
    for cycles in range(100):
        dist=[[]]*300
        dist=np.array(dist)
        for f in range(s):
            dist1=[]
        
            for p in range(len(data)):
                ed =  math.sqrt((kupdate[f][0]-datanew[p][0])**2 + (kupdate[f][1]-datanew[p][1])**2)
                dist1.append(ed)
            dist1=np.array(dist1)
            dist1=dist1.reshape(300,1)
            dist=np.append(dist,dist1,axis=1)


        min1 =np.argmin(dist,axis=1)
        matf=[]
        for q in range(s):    
            mat1=[]
            for i in range(len(data)):
                if q ==min1[i]:
                    mat1.append(data[i])
            matf.append(mat1)
        knew=[]
        for dimen in range(s):
            x=0
            y=0
            if len(matf[dimen])==0:
                t1=kupdate[dimen]
                knew.append(t1)
            else:
                for cent in range(len(matf[dimen])):
                    x= x+matf[dimen][cent][0]
                    y=y+matf[dimen][cent][1]
                x1=x/len(matf[dimen])
                y1=y/len(matf[dimen])
                t1=[x1,y1]
                knew.append(t1)
        if c==0:
            kupdate=knew
            c=c+1
            continue
        elif np.array_equal(knew,kupdate)==True:
            break
        else:
            kupdate=knew
            c=c+1
    
    c1=0
    for v1 in range(s):
        for v2 in range(len(matf[v1])):
            c1= c1 + (matf[v1][v2][0]-knew[v1][0])**2 + (matf[v1][v2][1]-knew[v1][1])**2
    fig=plt.figure(figsize=(20,10))
    color=['red','skyblue','blue','magenta','pink','orange','green','brown','yellow','black']
    leg=['k1 cluster','k2 cluster','k3 cluster','k4 cluster','k5 cluster','k6 cluster','k7 cluster','k8 cluster','k9 cluster','k10 cluster']
    leg2=['k1 center','k2 center','k3 center','k4 center','k5 center','k6 center','k7 center','k8 center','k9 center','k10 center']
    plot=plt.subplot(111)
    for w in range(s):
        arr=np.array(matf[w])
        plot.scatter(arr[:,0],arr[:,1],color=color[w],label=leg[w],s=5)
        plot.scatter(knew[w][0],knew[w][1],color=color[w],label=leg2[w],marker='x',s=200)
    chartBox = plot.get_position()
    plot.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
    plot.legend(loc='upper center', bbox_to_anchor=(1.15, 0.8), shadow=True, ncol=1)
    plt.title("clustered data with 'k++' initial clusters")
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.show()
    print('cost function of these clusters =',c1)
    print('number of cycles required for convergence =',c)
else:
    print('Enter a number between 2 and 10')


# In[8]:


# STRATEGY 2 COST FUNCTION GRAPH 
s=10
cfunc=[]
# Here we are using the same strategy as in K++ to generate the remaining clusters, with the first one being randomly generated
k = datanew[np.random.choice(datanew.shape[0], size=1, replace=False), :]
dist=[[]]*300
dist=np.array(dist)
# The only difference is that we generate all 9 remaining clusters so that we can calculate the corresponding cost function.
kj=np.array(k)
for size in range(s-1): # K++ Algorithm starts
    dist1=[]
    for p in range(len(data)):
        ed =  math.sqrt((kj[size][0]-datanew[p][0])**2 + (kj[size][1]-datanew[p][1])**2)
        dist1.append(ed)
    dist1=np.array(dist1)
    dist1=dist1.reshape(300,1)
    dist=np.append(dist,dist1,axis=1)
    if size==0:
        k2=np.array(datanew[np.argmax(dist,axis=0)])
    else:
        smat=np.array(np.sum(dist,axis=1))
        smat2=smat/len(kj)
        smat2=smat2.reshape(300,1)
        k2=np.array(datanew[np.argmax(smat2,axis=0)])
    kj=np.append(kj,k2,axis=0)

for z in range(2,s+1): # K-Means Algorithm runs for 9 times, with inital centers generated by K++ algorithm
    
    kupdate=kj
    c=0
    

    for cycles in range(30):
        dist=[[]]*300
        dist=np.array(dist)
        for f in range(z):
            dist1=[]
        
            for p in range(len(data)):
                ed =  math.sqrt((kupdate[f][0]-datanew[p][0])**2 + (kupdate[f][1]-datanew[p][1])**2)
                dist1.append(ed)
            dist1=np.array(dist1)
            dist1=dist1.reshape(300,1)
            dist=np.append(dist,dist1,axis=1)


        min1 =np.argmin(dist,axis=1)
        matf=[]
        for q in range(z):    
            mat1=[]
            for i in range(len(data)):
                if q ==min1[i]:
                    mat1.append(data[i])
            matf.append(mat1)
        knew=[]
        for dimen in range(z):
            x=0
            y=0
            if len(matf[dimen])==0:
                t1=kupdate[dimen]
                knew.append(t1)
            else:
                for cent in range(len(matf[dimen])):
                        x= x+matf[dimen][cent][0]
                        y=y+matf[dimen][cent][1]
                x=x/len(matf[dimen])
                y=y/len(matf[dimen])
                t1=[x,y]
                knew.append(t1)
        kupdate=knew
    c1=0
    for v1 in range(z):
        for v2 in range(len(matf[v1])):
            c1= c1 + (matf[v1][v2][0]-knew[v1][0])**2 + (matf[v1][v2][1]-knew[v1][1])**2
    cfunc.append(c1)
clusters=[]
for t in range(2,s+1):
    clusters.append(t)
plt.plot(clusters, cfunc)
plt.title("Elbow curve of 'k++' initial clusters")
plt.xlabel("number of 'k' clusters")
plt.ylabel("Cost Function")
print('cost function values are', cfunc)


# In[ ]:




