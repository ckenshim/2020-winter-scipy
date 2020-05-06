#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from CourseworkLibrary import *
import numpy as np

#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    
    mean = realData.sum(axis=0)/theData.shape[0]

    # Coursework 4 task 1 ends here
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    noSamples = theData.shape[0]

    # Using the equation provided in lecture slides
    U = realData - np.tile(Mean(realData), (realData.shape[0], 1))
    covar = np.dot(U.T, U)/(noSamples - 1)

    # Note that it produces the same output as the numpy function cov (the rowvar transforms so columns represent variables and rows observations)
    n_cov = np.cov(realData, rowvar=0)

    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(theBasis, file_prefix):
    
    # Coursework 4 task 3 begins here

    for idx, basis in enumerate(theBasis):
        SaveEigenface(basis, file_prefix + str(idx) + ".jpg")

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # take the dot product of the mean centred image against the eigenvector
    for basis in theBasis:
        magnitudes.append(np.dot((np.array(theFaceImage) - np.array(theMean)), basis))

    # Coursework 4 task 4 ends here
    return np.array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags, file_prefix):
    
    # Coursework 4 task 5 begins here

    # Start off with just the mean face
    p_x = aMean
    SaveEigenface(np.array(p_x), file_prefix + "MeanFace.jpg")

    # For each eigenface
    for idx, basis in enumerate(aBasis):

        # Add the extra information from each basis and component in (then save it)
        p_x += np.dot(basis, componentMags[idx].T)

        SaveEigenface(p_x, file_prefix + str(idx) + ".jpg")


    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    real_data = np.array(theData)

    U = real_data - np.tile(Mean(real_data), (real_data.shape[0], 1))

    # Generate our small matrix
    small_vec = np.dot(U, U.T)

    # Get small matrix's eigvenvalues and eigenvectors
    w, v = np.linalg.eig(small_vec)

    # Locate the 0 valued eigenvalue and remove it since it does not add new information in a new dimension
    zero_loc = np.where(w==0)[0]
    w = np.delete(w, zero_loc)

    # Convert the eigenvalues from the small matrix to the big one
    big_eig = np.dot(U.T, v)

    # Remove the eigenvector with the eigenvalue == 0
    big_eig = np.delete(big_eig, zero_loc, axis=1)

    # Sort in to the correct order (with argsort), and normalise each vector
    for item in argsort(w)[::-1]:
        orthoPhi.append(big_eig[:,item]/np.sqrt(np.sum(np.power(big_eig[:,item],2))))
    
    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# main program part for Coursework 1
#
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = np.array(datain)
AppendString("results.txt","Coursework Four Results by txl11")
AppendString("results.txt","") #blank line

AppendString("results.txt","Mean vector of the HepatitisC data set")
AppendList("results.txt",Mean(theData)) 

AppendString("results.txt","Covariance matrix of the HepatitisC data set")
AppendArray("results.txt",Covariance(theData)) 

print ('Saving Eigenfacefiles as PrincipalComponent_n.jpg')
CreateEigenfaceFiles(ReadEigenfaceBasis(), 'PrincipalComponent_')

projected_faces = ProjectFace(ReadEigenfaceBasis(), ReadOneImage('MeanImage.jpg'), ReadOneImage('c.pgm'))
AppendString("results.txt","Component Magnitudes for c.pgm")
AppendList("results.txt", projected_faces)

print ('Saving partial reconstructions as Reconstruction_n.jpg')
CreatePartialReconstructions(ReadEigenfaceBasis(), ReadOneImage('MeanImage.jpg'), projected_faces, "Reconstruction_")



our_basis = PrincipalComponents(ReadImages())

print ('Saving Eigenfacefiles from the basis computed from a-f.pgm as PrincipalComponentCustom_n')
CreateEigenfaceFiles(our_basis, 'PrincipalComponentCustom_')


projected_faces = ProjectFace(our_basis, Mean(array(ReadImages())), ReadOneImage('c.pgm'))

print ('Saving Eigenfacefiles from the basis computed from a-f.pgm as Reconstruction_Custom_n')
CreatePartialReconstructions(our_basis, Mean(array(ReadImages())), projected_faces, "Reconstruction_Custom_")
