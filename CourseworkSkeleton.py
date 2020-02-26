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
    noStates =  list(noStates)
    print(noStates)
    prior = np.zeros((noStates[root]), float) #prior = np.zeros((noStates[root]), float)

    for x in range(len(prior)):
        for i in theData:
            #print(root, x, i)
            if i[root]==x:
                prior[x]+=1
        prior[x] = prior[x]/len(theData)
    #print(prior)
            
        
    # Coursework 1 task 1 should be inserted here

    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    
    # Coursework 1 task 2 should be inserte4d here
    arrOf_0_ind = [i[0] for i in theData]
    #print(len(arrOf_0_ind))
    for i in range(len(arrOf_0_ind)):
        cPT[theData[i,varC]][theData[i,varP]]+=1
    #print(cPT)
    for j in range(noStates[varP]):
        sum_=(np.sum(theData[:,varP]==j))
        if sum_!=0:
            cPT[:,j]/=(np.sum(theData[:,varP]==j))
    #print(cPT)
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    arrOf_0_ind = [i[0] for i in theData]
    #print(len(arrOf_0_ind))
    for i in range(len(arrOf_0_ind)):
        jPT[theData[i,varRow]][theData[i,varCol]]+=1
    jPT/=len(arrOf_0_ind)
    #print(jPT)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    #print(aJPT)
    for i in range(len(aJPT[0])):
        sum_=(np.sum(aJPT[:,i]))
        if sum_!=0:
            aJPT[:,i]*=1/sum_
    #print(aJPT)
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    for i, el_rootPdf in enumerate(rootPdf):
        el_rootPdf=naiveBayes[0][i]
      
        for j in range(0,len(theQuery)):
            el_rootPdf*=naiveBayes[j][i]
            
    if (np.sum(rootPdf)!=0):
        rootPdf*=1/(np.sum(rootPdf))
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
    
    #print("------------------")
    #print(set(noStates))
    
    #print(theData[0])
    #print("------------------")
    AppendString("results.txt", "Coursework One Results by {0}".format("Galymzhan Abdimanap"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)
    #print(prior)

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The CPT probability 2/0")
    cpt = CPT(theData, 2, 0, noStates)
    AppendArray("results.txt", cpt)
    
    

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The JPT probability 2/0")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("results.txt", jPT)

    

    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The JPT2CPT probability")
    aJPT = JPT2CPT(jPT)
    AppendArray("results.txt", aJPT)



    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "Bayesian network")
    theQuery = noStates
    rootPdf = Query(theQuery, theData)
    AppendList("results.txt", rootPdf)


    # Continue filling the results.txt file here ...
