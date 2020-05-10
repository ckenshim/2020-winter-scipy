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
    for data in theData:
        prior[data[root]] += 1
    prior /= len(theData)
    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
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
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    # rootPdf = np.zeros((naiveBayes.shape[0]), float)
    cpt_table = [1, 2, 3, 4, 5]
    for i in range(len(theQuery)):
        cpt_table[i] = CPT(theData, i+1, 0, noStates)
    # print(cpt_table)

    for root in range(len(rootPdf)):
        rootPdf[root] = naiveBayes[root]
        for index, child in enumerate(theQuery):
            # print("Root: ", root, 'Query: ', child)
            # print(cpt_table[root][child][root])
            rootPdf[root] *= cpt_table[index][child][root]

    if np.sum(rootPdf) != 0:
        rootPdf = rootPdf / (np.sum(rootPdf))
    # print('Root Pdf: ', rootPdf)
    # end of coursework 1 task 5
    return rootPdf


#
# End of Coursework 1
#
# Coursework 2 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    mean = np.zeros(noVariables, float)
    # print('Real data', realData)
    # print('No variables: ', noVariables)
    # Coursework 2 task 1 begins here
    for row in realData:
        for index, value in enumerate(row):
            mean[index] += value
    mean /= realData.shape[0]
    # print('Mean: ', mean)

    # M = np.mean(realData.T, axis=1)
    # print('Mean with numpy: ', M)
    # Coursework 2 task 1 ends here
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables = theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 2 task 2 begins here
    I = realData - Mean(theData)
    covar = (1 / (realData.shape[0] - 1)) * np.dot(np.transpose(I), I)
    # covar = np.cov(realData.T)
    # Coursework 2 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis, path):
    # Coursework 2 task 3 begins here
    for index, value in enumerate(theBasis):
        filename = path + '/Eigenface' + str(index) + '.png'
        SaveEigenface(value, filename)
    # Coursework 2 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 2 task 4 begins here
    I = np.array(theFaceImage) - np.array(theMean)
    for i in theBasis:
        magnitudes.append(np.dot(I, i))
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags, path):
    # Coursework 2 task 5 begins here
    image = aMean
    for index, value in enumerate(aBasis):
        filename = path + '/partial' + str(index) + '.png'
        image += value * componentMags[index]
        SaveEigenface(image, filename)
    # Coursework 2 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    centeredData = theData - Mean(theData)
    UUT = np.dot(centeredData, np.transpose(centeredData))
    w, v = np.linalg.eig(UUT)

    zero_matrix = []
    for i, value in enumerate(w):
        if value == 0:
            zero_matrix.append(i)
    w = np.delete(w, zero_matrix)

    UTV = np.dot(np.transpose(centeredData), v)
    matrix_utv = np.delete(UTV, zero_matrix, axis=1)
    for i in np.flip(np.argsort(w)):
        orthoPhi.append(matrix_utv[:, i] / np.sqrt(np.sum(matrix_utv[:, i] ** 2)))

    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
#

if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = np.array(datain)
    AppendString("results.txt", "Coursework One Results by {0}".format("Tukpetov Raiymbet"))
    AppendString("results.txt", "")  # blank line
    AppendString("results.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("results.txt", prior)

    AppendString("results.txt", "The mean vector:")
    mean = Mean(theData)
    AppendList("results.txt", mean)

    AppendString("results.txt", "The covariance:")
    covariance = Covariance(theData)
    AppendArray("results.txt", covariance)

    theBasis = ReadEigenfaceBasis()
    CreateEigenfaceFiles(theBasis, 'eigenFaceFiles')

    meanFace = ReadOneImage('images/MeanImage.jpg')
    faceImage = ReadOneImage('images/c.pgm')
    magnitudes = ProjectFace(theBasis, meanFace, faceImage)
    AppendString("results.txt", "The magnitudes of c.pgm: ")
    AppendList("results.txt", magnitudes)

    CreatePartialReconstructions(theBasis, meanFace, magnitudes, 'partialReconstructions')

    images = np.array(ReadImages())
    newBasis = PrincipalComponents(images)
    CreateEigenfaceFiles(newBasis, 'NewBasis')
    newBasisMean = Mean(images)
    newBasisMags = ProjectFace(newBasis, newBasisMean, faceImage)
    CreatePartialReconstructions(newBasis, newBasisMean, newBasisMags, 'NewBasis')
    # Continue filling the results.txt file here ...
