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
    #noStatesset = list(noStates)
    
    prior = np.zeros((noStates[root]), float)
    # Coursework 1 task 1 should be inserted here
    for i in range(len(prior)):
        for j in range(len(theData)):
            if theData[j][i]==i:
                prior[i]+=1
    # end of Coursework 1 task 1
    prior=prior/len(theData)
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    for i in range(len(theData)):
        cPT[theData[i,varC]][theData[i,varP]]+=1

    sums=[]
    for i in range(noStates[varP]):
        sums.append(sum(cPT[:,i]))

    for i in range(len(cPT)):
        cPT[i]/=sums
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    for i in range(len(theData)):
        jPT[theData[i,varRow]][theData[i,varCol]]+=1
    jPT/=len(theData)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    sums=[]
    for i in range(len(aJPT[0])):
        sums.append(sum(aJPT[:,i]))

    for i in range(len(aJPT)):
        for j in range(len(aJPT[i])):
            aJPT[i][j]*=1/sums[j]
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    # Coursework 1 task 5 should be inserted here

    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("C:/Users/Kuanysh/Desktop/2020-winter-scipy-master/coursework1/data.txt")
    theData = np.array(datain)
    #theData = [list(x) for x in theData]
    #noStates = list(noStates)
    AppendString("results.txt", "Coursework One Results by {0}".format("Your Name"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

    # Continue filling the results.txt file here ...
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Conditional probability of node 0 and 2")
    cpt = CPT(theData, 2, 0, noStates)
    AppendArray("results.txt", cpt)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Joint probability of node 0 and 2")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", jPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "JPT2CPT")
    ajPT = JPT2CPT(jPT)
    AppendArray("results.txt", ajPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Query")
