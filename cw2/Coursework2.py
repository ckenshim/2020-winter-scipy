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
    noVariables=theData.shape[1]   # кол-во столбцов
    mean = np.mean(realData, axis = 0) # среднее значение по столбцам 
    
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    k = theData.shape[0]
    U = realData - np.tile(Mean(realData), (realData.shape[0], 1))
    covar = np.dot(U.T, U) / (k - 1)
    # Coursework 2 task 2 begins here

    
    # Coursework 2 task 2 ends here
    return covar

def CreateEigenfaceFiles(basis, fl="PrincipalComponent_"):
    # Coursework 2 task 3 begins here...
    fl_name = ''.join((fl, "{0}.jpg"))
    for i, component in enumerate(basis):
        SaveEigenface(component, fl_name.format(i))

    # Coursework 2 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 2 task 4 begins here
    faceImageData = ReadOneImage(theFaceImage)
    magnitudes = np.dot((faceImageData - theMean), np.transpose(theBasis))
    
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags, f = ''):
	for i in range(0, len(componentMags)):
		rec = np.add(np.dot(np.transpose(aBasis[0:i]), componentMags[0:i]), aMean)
		SaveEigenface(rec, f + "PartialReconstruction_" + str(i + 1) + ".jpg")
	
     #delete this when you do the coursework
    # Coursework 2 task 5 begins here

    # Coursework 2 task 5 ends here
    
def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    mean = np.matrix(theData.astype(float) - Mean(theData))
    UT = mean * mean.transpose()
    eignevalues, eigenvectors = np.linalg.eig(UT)
    phi = mean.transpose() * eigenvectors
    phi = phi / np.apply_along_axis(np.linalg.norm, 0, phi)
    ind = np.argsort(eignevalues)[::-1]
    orthoPhi = phi.transpose()[ind]
    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
#
if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = np.array(datain)
    mean_matrix = Mean(theData)
    covariance = Covariance(theData)
    basis = ReadEigenfaceBasis()
    print(basis)
    CreateEigenfaceFiles(basis)
    meanimage = np.array(ReadOneImage("MeanImage.jpg"))
    projected_magnitudes = ProjectFace(basis, meanimage, "c.pgm")
    CreatePartialReconstructions(basis, meanimage, projected_magnitudes)
    image_data = np.array(ReadImages())
    new_basis = PrincipalComponents(image_data)
    CreateEigenfaceFiles(new_basis, "new")
    new_mean = Mean(image_data)
    new_projected = ProjectFace(new_basis, new_mean, "c.pgm")
    CreatePartialReconstructions(new_basis, new_mean, new_projected, "new")

    AppendString("results2.txt", "Coursework One Results by {0}".format("Dinara_Baimagambetova"))
    AppendString("results2.txt", "")  # blank line
    AppendString("results2.txt", "The Mean vector HepatitisC dataset")
    AppendList("results2.txt", mean_matrix)
    AppendString("results2.txt", "The covariance matrix")
    AppendArray("results2.txt", covariance)
   
    AppendString("results2.txt", "The component magnitudes for image “c.pgm” in the principal component")
    AppendList("results2.txt", projected_magnitudes)
    AppendList("results2.txt", projected_magnitudes)
    AppendString("results2.txt", "New projected magnitudes")
    AppendList("results2.txt", new_projected)

    # Continue filling the results.txt file here ...

