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
    for i in range(len(prior)):
        for j in range(len(theData)):
            if theData[j][i] == i:
                prior[i] += 1
    prior = prior / len(theData)
    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    for i in range(len(theData)):
        cPT[theData[i, varC]][theData[i, varP]] += 1

    sums = []
    for i in range(noStates[varP]):
        sums.append(sum(cPT[:, i]))

    for i in range(len(cPT)):
        cPT[i] /= sums
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    for i in range(len(theData)):
        jPT[theData[i, varRow]][theData[i, varCol]] += 1
    jPT /= len(theData)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    sums = []
    for i in range(len(aJPT[0])):
        sums.append(sum(aJPT[:, i]))

    for i in range(len(aJPT)):
        for j in range(len(aJPT[i])):
            aJPT[i][j] *= 1 / sums[j]
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes.shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    prior = Prior(theData, 0, noStates)
    cpt1 = CPT(theData, 1, 0, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT(theData, 3, 0, noStates)
    cpt4 = CPT(theData, 4, 0, noStates)
    cpt5 = CPT(theData, 5, 0, noStates)
    table = [prior, cpt1, cpt2, cpt3, cpt4, cpt5]

    for i in range(len(naiveBayes)):
        rootPdf[i] = prior[i]
        for j, query in enumerate(theQuery):
            cpt = table[j + 1]
            rootPdf[i] *= cpt[query, i]

    rootPdf = rootPdf / sum(rootPdf)
    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#


if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("data.txt")
    print(datain)
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Nuradin Islam"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

    # Continue filling the results.txt file here ...
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Conditional probability of node 0 and 2")
    cPT = CPT(theData, 2, 0, noStates)
    AppendArray("results.txt", cPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Joint probability of node 0 and 2")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", jPT)

    AppendString("results.txt", "The Joint probability to The Conditional probability")
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "JPT2CPT")
    ajPT = JPT2CPT(jPT)
    AppendArray("results.txt", ajPT)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Query of [4,0,0,0,5]")
    rootPdf = Query([4, 0, 0, 0, 5], prior)
    AppendList("results.txt", rootPdf)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The Query of [6, 5, 2, 5, 5]")
    rootPdf = Query([6, 5, 2, 5, 5], prior)
    AppendList("results.txt", rootPdf)
