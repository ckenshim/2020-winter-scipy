#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Coursework in Python 
from CourseworkLibrary import *
import numpy as np
from numpy import linalg as LA

#
# Coursework 1 begins here
#
def Prior(theData, root, noStates):
    prior = np.zeros((noStates[root]), float)
    # Coursework 1 task 1 should be inserted here
    unique_elements, counts_elements = np.unique(theData[:,root], return_counts=True)
    prior = counts_elements/len(theData[:,root])

    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    # A - VarP  B - varC ,p(B|A) = P(A&B)/P(A)
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    P = Prior(theData, varP, noStates)
    datC = theData[:,varC]
    datP = theData[:,varP]
    #P(A&B)
    for i in range(len(datC)):
        cPT[datC[i],datP[i]] = cPT[datC[i],datP[i]] +1

    cPT = cPT/np.sum(cPT)
    #p(B|A)
    for i in range(noStates[varC]):
        for j in range(noStates[varP]):
            cPT[i,j] = cPT[i,j]/P[j]
    #end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varCol, varRow, noStates):
    jPT = np.zeros((noStates[varCol], noStates[varRow]), float)
    # Coursework 1 task 3 should be inserted here
    datC = theData[:,varRow]
    datP = theData[:,varCol]
    for i in range(len(datP)):
        jPT[datP[i],datC[i]] = jPT[datP[i],datC[i]] +1
    jPT = jPT/np.sum(jPT)
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    N = len(aJPT[:,0])
    M = len(aJPT[0,:])
    arr = np.zeros((N,M),float)
    for i in range(0,M):
        pa = np.sum(aJPT[:,i])
        for j in range(0,N):
            arr[j,i] = aJPT[j,i]/pa    
    # coursework 1 task 4 ends here
    aJPT = arr
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes.shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    # p(A|B,C,D,E) = alpha * P(A)*P(B|A)*P(C|A)*P(D|A)*P(E|A)
    for k in range(len(naiveBayes)):
        rootPdf[k] = prior[k] 
        j = 1
        for i in theQuery:
            cond = CPT(theData,j,0,noStates)
            rootPdf[k] = rootPdf[k]*cond[i,k]
            j = j+1
    
    alpha = 1/np.sum(rootPdf)
    rootPdf = alpha*rootPdf
    return rootPdf


#
# End of Coursework 1
#


# Coursework 2 begins here
#
def Mean(theData):
    theData = np.array(theData)
    realData = theData.astype(float)
    
    # Coursework 2 task 1 begins here
    N = theData.shape[0]
    mean = np.sum(realData,axis=0)/N
    # Coursework 2 task 1 ends here
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
 
    # Coursework 2 task 2 begins here
    U = realData-Mean(realData)
    N = realData.shape[0]
    covar = np.dot(U.T,U)/(N-1)

    # Coursework 2 task 2 ends here
    return covar


def CreateEigenfaceFiles(theBasis,path):

    # Coursework 2 task 3 begins here
    j = 0
    for i in theBasis:
        filename = path+"\\PrincipalComponent" + str(j) + ".jpg" 
        SaveEigenface(i, filename)
        j+=1
    # Coursework 2 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
 
    # Coursework 2 task 4 begins here
    for i in theBasis:
        magnitudes.append(np.dot((np.array(theFaceImage) - np.array(theMean)), i))
    
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)


def CreatePartialReconstructions(aBasis, aMean, componentMags,path):
    # Coursework 2 task 5 begins here
    
    filename = path+ "\\MeanFace.jpg" 
    Partial = aMean
    SaveEigenface(np.array(aMean), filename)
    i = 0
    for basis in aBasis:

        filename = path+"\\Reconstruction" + str(i) + ".jpg" 

        Partial += np.dot(basis, componentMags[i].T)
        
        SaveEigenface(Partial, filename)
        i+=1
    # Coursework 2 task 5 ends here


def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    U = theData-Mean(theData)
  
    #print(U.shape)
    UUT = np.dot(U,U.T)
    
    w, v = LA.eig(UUT)

    idx = []
    for i in range(len(w)):
        if w[i] == 0:
            
            idx.append(i)
    w = np.delete(w,idx)

    UTV = np.dot(U.T,v)
    UTV = np.delete(UTV,idx,axis=1)
    indx = np.flip(np.argsort(w))  
    for i in indx:
        orthoPhi.append(UTV[:,i]/np.sqrt(np.sum(np.power(UTV[:,i],2))))
    # Coursework 4 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
#

if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = np.array(datain)
    AppendString("Results2.txt", "Coursework2 Results2 by {0}".format("Aydarken Yerkanat"))
    AppendString("Results2.txt","")

    AppendString("Results2.txt","Mean vector")
    AppendList("Results2.txt",Mean(theData)) 

    AppendString("Results2.txt","Covariance matrix")
    AppendArray("Results2.txt",Covariance(theData))

    DataImg = ReadImages()
    DataMean = Mean(DataImg)
    EigenfaceBasis = ReadEigenfaceBasis()

    CreateEigenfaceFiles(EigenfaceBasis,"eigen1")

    MeanImg = ReadOneImage('images\\MeanImage.jpg')
    ImgC = ReadOneImage('images\\c.pgm')
    ProjectFaces1 = ProjectFace(EigenfaceBasis, MeanImg, ImgC)

    AppendString("Results2.txt","Component Magnitudes for c.pgm")
    AppendList("Results2.txt", ProjectFaces1)

    CreatePartialReconstructions(EigenfaceBasis, MeanImg, ProjectFaces1,"partial1")

    NewBasis = PrincipalComponents(DataImg)
    
    
    CreateEigenfaceFiles(NewBasis,"eigen2")
    ProjectFace2 = ProjectFace(NewBasis, DataMean, ImgC)
    CreatePartialReconstructions(NewBasis, DataMean, ProjectFace2,"partial2")
    
