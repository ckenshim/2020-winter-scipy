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
    # print('Root: ', root)
    # print('States: ', noStates)
    # print('Prior first: ', prior)
    for data in theData:
        prior[data[root]] += 1
    # print('Prior second: ', prior)
    prior /= len(theData)
    # print('Prior third: ', prior)
    # print(sum(prior))
    # end of Coursework 1 task 1
    return prior

# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), dtype=np.float32)
    # Coursework 1 task 2 should be inserte4d here
    for data in theData:
        cPT[data[varC], data[varP]] += 1
    total = np.sum(cPT, axis=0)
    cPT = cPT / total
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    for data in theData:
        list_m = np.array(list(data))
        jPT[list_m[varRow], list_m[varCol]] += 1
    jPT /= len(theData)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    total = np.sum(aJPT, axis=0)
    aJPT = aJPT / total
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes, noStates):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    prior = Prior(naiveBayes, 0, noStates)
    # print('Query root prior:', prior)

    # for node in theQuery:
    rootPdf[0] = np.sum(prior)
    for child_node in range(1, len(theQuery)+1):
        cPT = CPT(naiveBayes, child_node, 0, noStates)
        cPTandPrior = cPT[theQuery[child_node-1]]*prior
        rootPdf[child_node] = np.sum(cPTandPrior)
        # print('CPT', child_node, '\n', cPT[theQuery[child_node-1]])
        # print('CPT and Prior', child_node, '\n', cPTandPrior)

    # print('Root Pdf: ', rootPdf)
    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("data.txt")
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Tukpetov Raiymbet"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    cPT = CPT(theData, 2, 0, noStates)
    jPT = JPT(theData, 2, 0, noStates)
    jPT2cPT = JPT2CPT(jPT)
    query = Query([4, 0, 0, 0, 5], theData, noStates)
    query2 = Query([6, 5, 2, 5, 5], theData, noStates)
    AppendList("results.txt", prior)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "Conditional probability matrix:")
    AppendArray("results.txt", cPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "Joint probability matrix::")
    AppendArray("results.txt", jPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "Conditional probability matrix calculated from the joint probability matrix:")
    AppendArray("results.txt", jPT2cPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The results of queries [4,0,0,0,5] on the naive network:")
    AppendList("results.txt", query)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The results of queries [6, 5, 2, 5, 5] on the naive network:")
    AppendList("results.txt", query2)

    print("The prior probability of node 0:", prior)
    print("Conditional probability matrix:\n", cPT)
    print("Joint probability matrix:\n", jPT)
    print("Conditional probability matrix calculated from the joint probability matrix:\n", jPT2cPT)
    print("The results of queries [4,0,0,0,5] on the naive network:\n", query)
    print("The results of queries [6, 5, 2, 5, 5] on the naive network:\n", query2)
    # Continue filling the results.txt file here ...
