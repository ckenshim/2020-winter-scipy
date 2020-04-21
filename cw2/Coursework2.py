#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Coursework in Python 
import sys
import os
from operator import itemgetter
from CourseworkLibrary import *
import numpy as np
from pprint import pprint


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
    noVariables=theData.shape[1]
#     print(noVariables)
    mean = []
    # Coursework 2 task 1 begins here
    mean = np.sum(theData,axis=0)/theData.shape[0]

    # Coursework 2 task 1 ends here
    return np.array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    covar = np.zeros((noVariables, noVariables), float)
    # Coursework 2 task 2 begins here
    t = theData.shape[0]
    
    U = realData - np.tile(Mean(realData), (realData.shape[0], 1))
    covar = np.dot(U.T, U)/(t - 1)

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
    for i in theBasis:
        magnitudes.append(np.dot((np.array(theFaceImage) - np.array(theMean)), i))
    # Coursework 2 task 4 ends here
    return np.array(magnitudes)

def CreatePartialReconstructions(basis, mean, magnitudes, fl="PartialReconstruction"):
    # Coursework 2 task 5 begins here...
    b_matrix = np.matrix(basis)
    len_mag = len(magnitudes)
    fl_name = ''.join((fl, "{0}.jpg"))
    for i in range(len_mag + 1):
        m_matrix = np.matrix(magnitudes * ([1] * i + [0] * (len_mag - i)))
        SaveEigenface(np.array((m_matrix * b_matrix + mean))[0],
                            fl_name.format(i))
    # Coursework 2 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 2 task 6 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    theData_array = np.array(theData)
#     print(theData_array)
    U = theData_array - np.tile(Mean(theData_array), (theData_array.shape[0], 1))
    
    dot_U = np.dot(U, U.T)
    w, v = np.linalg.eig(dot_U)
    zero = np.where(w==0)[0]
    w = np.delete(w, zero)
    dot_UT = np.delete(np.dot(U.T, v), zero, axis=1)
    for item in np.argsort(w)[::-1]:
        orthoPhi.append(dot_UT[:,item]/np.sqrt(np.sum(np.power(dot_UT[:,item],2))))
    
    # Coursework 2 task 6 ends here
    return np.array(orthoPhi)

#
# End of Coursework 2
#

if __name__ == '__main__':
    noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
    theData = np.array(datain)
    
    fl = "Results2.rst"
    if os.path.exists(fl):
        os.remove(fl)
    AppendString(fl, "Coursework 2 Results by {0}".format("Zhansultan"))
    AppendString(fl, "")  # blank line
    
    AppendString(fl,"function Mean results: ")
    AppendList(fl,Mean(theData))
    
    AppendString("Results2.txt", "")
    AppendString("Results2.txt", "function Mean results:")
    AppendList("Results2.txt", Mean(theData))
    
    AppendString(fl,"function Covariance results:")
    AppendArray(fl,Covariance(theData))
    AppendString("Results2.txt", "")
    AppendString("Results2.txt", "function Covariance results:")
    AppendArray("Results2.txt", Covariance(theData))
    
    basis = np.array(ReadEigenfaceBasis())
    CreateEigenfaceFiles(basis)
    for i in range(len(basis)):
        AppendString(fl, ''.join(("splitimage PrincipalComponent_", str(i), ".jpg")))
    
    mean_face = np.array(ReadOneImage("images/MeanImage.jpg"))
    face = np.array(ReadOneImage('images/c.pgm'))
    magnitudes = ProjectFace(basis, mean_face, face)
    AppendList(fl, magnitudes)
    AppendString(fl, "")
    
    CreatePartialReconstructions(basis, mean_face, magnitudes)
    for i in range(len(basis) + 1):
        AppendString(fl, ''.join(("splitimage PartialReconstruction", str(i), ".jpg")))
    
    image_data = np.array(ReadImages())
    new_basis = PrincipalComponents(image_data)
    CreateEigenfaceFiles(new_basis, 'NewPrincComp')
    for i in range(len(new_basis)):
        AppendString(fl, ''.join(("splitimage NewPrincComp", str(i), ".jpg")))
    
    new_mean = Mean(image_data)
    face2 = np.array(ReadOneImage('images/c.pgm'))
    new_magnitudes = ProjectFace(new_basis, new_mean, face2)
    AppendString(fl, "The magnitudes of 'c.pgm' from given basis:")
    AppendList(fl, new_magnitudes)
    CreatePartialReconstructions(new_basis, new_mean, new_magnitudes, "NewPartRecon")
    
    for i in range(len(new_basis) + 1):
        AppendString(fl, ''.join(("splitimage NewPartRecon",str(i), ".jpg")))
    
    
    os.system("rst2latex.py {0}.rst {0}.tex {0}.pdf".format(fl.rpartition('.')[0]))
    os.system("pdflatex {0}.tex".format(fl.rpartition('.')[0]))
    
    # Continue filling the results.txt file here ...


# Загрузка фото в PDF
from fpdf import FPDF 
import re
pdf = FPDF()    
def add_image(image_path):
#     print(image_path)
    pdf.image(image_path, w=pdf.w/5.0, h=pdf.h/5.0)
    pdf.ln(8)
 
    # Image caption
    pdf.cell(3.0, 0.0, image_path)
    pdf.ln(15)
    
pdf.add_page() 

pdf.set_font("Arial", size = 15) 

f = open("Results2.rst", "r")

    
for x in f:
#     print(x)
    if re.search(r'\bsplitimage\b', x):
#         print(x)
#         print('---------------')
        splitImageName = x.replace("splitimage ", "")
        splitImageName2 = splitImageName.replace("\n", "")
#         print(splitImageName2)
        add_image(splitImageName2)
#         add_image("'"+ splitImageName2 + "'")
    else:
        pdf.cell(200, 10, txt = x, ln = 1, align = 'C') 
#     pdf.cell(200, 10, txt = x, ln = 1, align = 'C') 
pdf.output("Results2.pdf")  
