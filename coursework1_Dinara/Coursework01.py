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
    rootValues = theData[:,root]
    occurences = np.bincount(rootValues)
    occTotal = sum(occurences)
    prior = np.array(occurences, float) / occTotal
    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    CValues = theData[:,varC]
    PValues = theData[:,varP]
    CStates = noStates[varC]
    PStates = noStates[varP]
    COcc = np.bincount(PValues)
    for i in range(CStates):
        for j in range(PStates):
            joint = len([k for k in range(len(theData)) if CValues[k] == i and PValues[k] == j])
            cPT[i][j] = float(joint) / COcc[j]
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    rowValues = theData[:,varRow]
    colValues = theData[:,varCol]
    dataPoints = len(theData)
    rowStates = noStates[varRow]
    colStates = noStates[varCol]
    for i in range(rowStates):
        for j in range(colStates):
            joint = len([x for x in range(len(theData)) if rowValues[x] == i and colValues[x] == j])
            jPT[i][j] = float(joint) / dataPoints
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    cols = aJPT.shape[1]
    for i in range(cols):
        currentCol = aJPT[:, i]
        alpha = float(1) / sum(currentCol)
        aJPT[:, i] = alpha * currentCol
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    evidence = 1
    for i in range(len(theQuery)):
        evidence *= naiveBayes[i+1][theQuery[i],:]
    rootPdf = np.transpose(naiveBayes[0]) * evidence
    alpha = float(1) / sum(rootPdf)    
    rootPdf = alpha * rootPdf
    # end of coursework 1 task 5
    return rootPdf

def createNetwork(theData, noVariables, noRoots, noStates):
    naiveBayes = []
    for i in range(noRoots):
        naiveBayes.append(Prior(theData, i, noStates))
    noNonRoots = noVariables - noRoots    
    for i in range(noNonRoots):
        var = i + noRoots
        naiveBayes.append(CPT(theData, var, 0, noStates))
    return np.array(naiveBayes)
#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("data.txt")
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Dinara"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "1. The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

    # Continue filling the results.txt file here ...
    AppendString("results.txt","2. The conditional probability matrix P (2|0) calculated from the data")
    cPT = CPT(theData, 2, 0, noStates)
    AppendArray("results.txt", cPT)


    AppendString("results.txt","3. The joint probability matrix P (2&0) calculated from the data")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", jPT)

    AppendString("results.txt","4. The conditional probability matrix P (2|0) calculated from the joint probability matrix P (2&0)")
    convertedCPt = JPT2CPT(jPT)
    AppendArray("results.txt", convertedCPt)

    naiveBayes = createNetwork(theData, noVariables, noRoots, noStates)

    AppendString("results.txt"," The resultss of queries [4,0,0,0,5] and [6, 5, 2, 5, 5] on the naive network")
    AppendString("results.txt","The resultss of [4,0,0,0,5] on the naive network")
    query = Query([4,0,0,0,5], naiveBayes)
    AppendList("results.txt", query)
    AppendString("results.txt","The resultss of [6, 5, 2, 5, 5] on the naive network")
    query = Query([6, 5, 2, 5, 5], naiveBayes)
    AppendList("results.txt", query)

