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
    #print(aJPT)
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
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
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
# Coursework 2 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    mean = []
    # Coursework 2 task 1 begins here
    mean = np.sum(theData, axis=0)/theData.shape[0]
    # Coursework 2 task 1 ends here
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 2 task 2 begins here
    covar = np.cov(theData.T)
    # Coursework 2 task 2 ends here
    return covar


def CreateEigenfaceFiles(theBasis):
    # Coursework 2 task 3 begins here
    # theBasis = [list(basis) for basis in theBasis]
    for i, el in enumerate(theBasis):
        SaveEigenface(np.array(el), f"eigenfaces/EFmy_{str(i)}.jpg")
    # Coursework 2 task 3 ends here


def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 2 task 4 begins here
    # theBasis = [list(basis) for basis in theBasis]
    for basis in theBasis:
        magnitudes.append(np.dot((np.array(theFaceImage) - np.array(theMean)), basis))
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)


def CreatePartialReconstructions(aBasis, aMean, componentMags):
    # Coursework 2 task 5 begins here

    tempcomponent = aMean
    SaveEigenface(np.array(aMean), "recons/MeanFacemy.jpg")

    for i, basis in enumerate(aBasis):
        tempcomponent = np.add(tempcomponent, np.dot(basis, componentMags[i].T))
        SaveEigenface(tempcomponent, "recons/recmy"+str(i)+".jpg")

    # Coursework 2 task 5 ends here


def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    theData_array = np.array(theData)
    U = theData_array - np.tile(Mean(theData_array), (theData_array.shape[0], 1))

    dot_U = np.dot(U, U.T)
    w, v = np.linalg.eig(dot_U)
    zero = np.where(w==0)[0]
    w = np.delete(w, zero)
    dot_UT = np.delete(np.dot(U.T, v), zero, axis=1)
    for item in np.argsort(w)[::-1]:
        orthoPhi.append(dot_UT[:,item]/np.sqrt(np.sum(np.power(dot_UT[:,item],2))))
    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
#


if __name__ == '__main__':

    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = np.array(datain)
    # theBasis = ReadEigenfaceBasis()
    theBasis = [list(basis) for basis in ReadEigenfaceBasis()]

    AppendString("results2.txt", "Coursework Two Results by {0}".format("Nuradin Islam"))
    AppendString("results2.txt", "")  # blank line
    AppendString("results2.txt", "Mean:")
    AppendList("results2.txt", Mean(theData))

    AppendString("results2.txt", "")
    AppendString("results2.txt", "Covariance Matrix:")
    AppendArray("results2.txt", Covariance(theData))

    CreateEigenfaceFiles(theBasis)

    theMean = np.array(ReadOneImage("images/MeanImage.jpg"))
    theFaceImage = np.array(ReadOneImage('images/c.pgm'))
    componentMags = ProjectFace(theBasis, theMean, theFaceImage)
    CreatePartialReconstructions(theBasis, theMean, componentMags)


    imageData = np.array(ReadImages())
    pgm_Basis = PrincipalComponents(imageData)

    CreateEigenfaceFiles(pgm_Basis)

    componentMags = ProjectFace(pgm_Basis, Mean(imageData), theFaceImage)
    CreatePartialReconstructions(pgm_Basis, Mean(imageData), componentMags)
