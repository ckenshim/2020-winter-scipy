#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Coursework in Python 
from CourseworkLibrary import *
import numpy as np


#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = np.zeros((noStates[root]), float)
    # Coursework 1 task 1 should be inserted here
    data_amount=len(theData[:,0])
    for i in range(data_amount):
        prior[theData[i,root]]+=1
    prior=prior/data_amount

    return prior

    # end of Coursework 1 task 1
    


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    data_amount=len(theData[:,0])
    for row in range(data_amount):
        cPT[theData[row,varC]][theData[row,varP]]+=1
    for i in range(noStates[varP]):
        alpha=(np.sum(theData[:,varP]==i))
        if alpha!=0:
            cPT[:,i]=cPT[:,i]/(np.sum(theData[:,varP]==i))

    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    data_amount=len(theData[:,0])
    for row in range(data_amount):
        jPT[theData[row,varRow]][theData[row,varCol]]+=1
    jPT=jPT/data_amount
    # end of coursework 1 task 3
    return jPT

# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    for i in range(len(aJPT[0,:])):
        alpha=(numpy.sum(aJPT[:,i]))
        if alpha!=0:
            aJPT[:,i]=aJPT[:,i] * (1/alpha)
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    for i in range(len(rootPdf)):
        rootPdf[i]=naiveBayes[0][i]
      
        for j in range(0,len(theQuery)):
            rootPdf[i]=rootPdf[i]*naiveBayes[j+1][theQuery[j],i]
	
    if (np.sum(rootPdf)!=0):
        rootPdf*=1/(np.sum(rootPdf)) #shouldnt be 0
    else:
        rootPdf=np.ones((naiveBayes[0].shape[0]), float)/naiveBayes[0].shape[0]
    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("data.txt")
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Your Name"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

    # Continue filling the results.txt file here ...
