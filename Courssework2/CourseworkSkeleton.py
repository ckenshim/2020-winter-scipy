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

    noStates =  list(noStates)
    print(noStates)
    prior = np.zeros((noStates[root]), float) #prior = np.zeros((noStates[root]), float)

    for x in range(len(prior)):
        for i in theData:
            #print(root, x, i)
            if i[root]==x:
                prior[x]+=1
        prior[x] = prior[x]/len(theData)

    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here

    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here

    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here

    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here

    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#
# Coursework 2 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    mean = np.zeros(theData.shape[1], float)
    # Coursework 2 task 1 begins here
    for x in range(len(mean)):
        for i in theData:
            #print(x, i)
            mean[x]+=i[x]
        mean[x] = mean[x]/len(theData)


    # Coursework 2 task 1 ends here
    return mean


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 2 task 2 begins here
    for i in range(noVariables):
            for j in range(noVariables):
                covar[i][j] = np.sum((realData[i] - realData[i].mean())*(realData[j] - realData[j].mean()))/(len(realData[i]) - 1)


    # Coursework 2 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 2 task 3 begins here
    fl_name = ''.join(("img/PrincipalComponent_", "{0}.jpg"))
    for i, value in enumerate(theBasis):
        SaveEigenface(theBasis[i], fl_name.format(i))
    # Coursework 2 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 2 task 4 begins here
    sub = np.subtract(theFaceImage, theMean)
    for i in theBasis:
        dot = np.dot(sub, i)
        magnitudes.append(dot)
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)

def CreatePartialReconstructions(theBasis, theMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 2 task 5 begins here
    #print(len(componentMags))
    arr = np.zeros(len(componentMags), int)
    for i in range(len(componentMags)):
        m_matrix = np.matrix(componentMags * arr)
        #print(m_matrix)
        arr[i]=1
        SaveEigenface(np.array((m_matrix * theBasis + theMean))[0],"img/PartialReconstruction_"+str(i)+".jpg")
    # Coursework 2 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    #U = theData_array - np.tile(Mean(theData_array), (theData_array.shape[0], 1))
    U = theData - Mean(theData)
    
    U_dot_UT = np.dot(U, U.T)
    w, v = np.linalg.eig(U_dot_UT)
    #print("------------------------------------------------------------")
    #print(w)
    arr_for_0=[]
    for i,el in enumerate(w):
        if el==0:
            arr_for_0.append(i)
    #print("------------------------------------------------------------")
    #print(arr_for_0)
    w = np.delete(w, arr_for_0)
    UT_dot_v = np.delete(np.dot(U.T, v), arr_for_0, axis=1)
    for i in np.flip(np.argsort(w)):
        orthoPhi.append(UT_dot_v[:,i]/np.sqrt(np.sum(UT_dot_v[:,i]**2)))


    #print(orthoPhi)
    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
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
    theBasis = ReadEigenfaceBasis()
    theBasis = np.array(theBasis)
    print(f'asd',theBasis)
    AppendString("results.txt", "Function Mean results:")
    mean = Mean(theData)
    AppendList("results.txt", mean)

    AppendString("results.txt", "Function Covarience results:")
    covariance = Covariance(theData)
    AppendArray("results.txt", covariance)

    CreateEigenfaceFiles(theBasis)

    theMean = np.array(ReadOneImage("images/MeanImage.jpg"))
    theFaceImage = np.array(ReadOneImage('images/c.pgm'))
    componentMags = ProjectFace(theBasis, theMean, theFaceImage)

    CreatePartialReconstructions(theBasis, theMean, componentMags)

    image = np.array(ReadImages())
    newTheBasis = PrincipalComponents(image)
    CreateEigenfaceFiles(newTheBasis)
    newTheMean = Mean(image)
    newComponentMags = ProjectFace(newTheBasis, newTheMean, theFaceImage)
    CreatePartialReconstructions(newTheBasis, newTheMean, newComponentMags)