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
    unique_elements, counts_elements = np.unique(theData[:,root], return_counts=True)
    prior = counts_elements/len(theData[:,root])

    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    # A - VarP  B - varC ,p(B|A) = P(A&B)/P(A)
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    P = Prior(theData, varP, noStates)
    datC = theData[:,varC]
    datP = theData[:,varP]
    #P(A&B)
    for i in range(len(datC)):
        cPT[datC[i],datP[i]] = cPT[datC[i],datP[i]] +1

    cPT = cPT/np.sum(cPT)
    #p(B|A)
    for i in range(noStates[varC]):
        for j in range(noStates[varP]):
            cPT[i,j] = cPT[i,j]/P[j]
    #end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varCol, varRow, noStates):
    jPT = np.zeros((noStates[varCol], noStates[varRow]), float)
    # Coursework 1 task 3 should be inserted here
    datC = theData[:,varRow]
    datP = theData[:,varCol]
    for i in range(len(datP)):
        jPT[datP[i],datC[i]] = jPT[datP[i],datC[i]] +1
    jPT = jPT/np.sum(jPT)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    N = len(aJPT[:,0])
    M = len(aJPT[0,:])
    arr = np.zeros((N,M),float)
    for i in range(0,M):
        pa = np.sum(aJPT[:,i])
        for j in range(0,N):
            arr[j,i] = aJPT[j,i]/pa    
    # coursework 1 task 4 ends here
    aJPT = arr
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes.shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    # p(A|B,C,D,E) = alpha * P(A)*P(B|A)*P(C|A)*P(D|A)*P(E|A)
    for k in range(len(naiveBayes)):
        rootPdf[k] = prior[k] 
        j = 1
        for i in theQuery:
            cond = CPT(theData,j,0,noStates)
            rootPdf[k] = rootPdf[k]*cond[i,k]
            j = j+1
    
    alpha = 1/np.sum(rootPdf)
    rootPdf = alpha*rootPdf
    return rootPdf


#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("data.txt")
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Aydarken Yerkanat"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

     
    # Continue filling the results.txt file here ...
    CPT1 = CPT(theData, 2, 0, noStates)
    AppendString("results.txt", "The conditional probability table of nodes 2,0")
    AppendArray("results.txt", CPT1)

    AppendString("results.txt", "The joint probability table of nodes 2,0")
    JPT1 = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", JPT1)
    JPT2 = JPT2CPT(JPT1)

    AppendString("results.txt", "The conditional probability table from joint probability table of nodes 2,0")
    AppendArray("results.txt", JPT2)
    
    theQuery = np.array([4,0,0,0,5])
    theQuery1 = np.array([6, 5, 2, 5, 5])
    naiveBayes = prior
    
    Z1 = Query(theQuery, naiveBayes)    
    AppendString("results.txt", "The results of queries [4,0,0,0,5] on the naive network")
    AppendList("results.txt", Z1)

    z2 = Query(theQuery1, naiveBayes)
    AppendString("results.txt", "The results of queries [6, 5, 2, 5, 5]  on the naive network")
    AppendList("results.txt", z2)
'''    for i in range(len(theData)):
        z = theData[i,1:]
        print(Query(z, naiveBayes), '  -   ',theData[i,0]+1)'''
