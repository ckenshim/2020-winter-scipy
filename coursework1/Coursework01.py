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
    n=len(theData[:,0])
    for i in range(n):
        j=theData[i,root]
        prior[j]+=1
    prior/=n
    # end of Coursework 1 task 1
    return prior


# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = np.zeros((noStates[varC], noStates[varP]), float)
    # Coursework 1 task 2 should be inserte4d here
    n=len(theData[:,0])
    for i in range(n):
        c=theData[i,varC]
        p=theData[i,varP]
        cPT[c][p]+=1
    for j in range(noStates[varP]):
        a=(np.sum(theData[:,varP]==j))
        if a!=0:
            parent=theData[:,varP]
            cPT[:,j]/=(np.sum(parent==j))
    # end of coursework 1 task 2
    return cPT


# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = np.zeros((noStates[varRow], noStates[varCol]), float)
    # Coursework 1 task 3 should be inserted here
    n=len(theData[:,0])
    for i in range(n):
        ro=theData[i,varRow]
        co=theData[i,varCol]
        jPT[ro][co]+=1
    jPT/=n
    # end of coursework 1 task 3
    return jPT


# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
    # Coursework 1 task 4 should be inserted here
    n=len(aJPT[0,:])
    for i in range(n):
        a=(np.sum(aJPT[:,i]))
        if a!=0:
            aJPT[:,i]*=1/a
    # coursework 1 task 4 ends here
    return aJPT


# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes):
    rootPdf = np.zeros((naiveBayes[0].shape[0]), float)
    # Coursework 1 task 5 should be inserted here
    for i in range(len(rootPdf)):
        rootPdf[i]=naiveBayes[0][i]
      
        for j in range(0,len(theQuery)):
            rootPdf[i]*=naiveBayes[j][i]
            
    if (np.sum(rootPdf)!=0):
        rootPdf*=1/(np.sum(rootPdf)) #shouldnt be 0
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
#   1. A title giving your full name
    AppendString("Results01.txt", "Coursework One Results by {0}".format("Zhansultan"))
    
#   prior start
#   2. The prior probability distribution of node 0 in the data set
    AppendString("Results01.txt", "")  # blank line
    AppendString("Results01.txt", "The prior probability of node 0")
    prior = Prior(theData, 0, noStates)
    AppendList("Results01.txt", prior)
#   prior end

#   The CPT probability start
#   3. The conditional probability matrix P(2|0) calculated from the data.
    AppendString("Results01.txt", "")  # blank line
    AppendString("Results01.txt", "The CPT probability")
    cPT = CPT(theData, 2, 0, noStates)
    AppendArray("Results01.txt", cPT)
#   The CPT probability end
    
#   The JPT probability start
#   4. The joint probability matrix P(2&0) calculated from the data.
    AppendString("Results01.txt", "")  # blank line
    AppendString("Results01.txt", "The JPT probability")
    jPT = JPT(theData, 2, 0, noStates)
    AppendArray("Results01.txt", jPT)
#   The JPT probability end
    
#   The JPT2CPT probability start
#   5. The conditional probability matrix P(2|0) calculated from the joint probability matrix P(2&0) .
    AppendString("Results01.txt", "")  # blank line
    AppendString("Results01.txt", "The JPT2CPT probability")
    aJPT = JPT2CPT(jPT)
    AppendArray("Results01.txt", aJPT)
#   The JPT2CPT probability end

#   Query a naive Bayesian network start
#   6. The results of queries [4,0,0,0,5] and [6, 5, 2, 5, 5] on the naive network
    AppendString("Results01.txt", "")  # blank line
    AppendString("Results01.txt", "Query a naive Bayesian network")
    theQuery = prior
    rootPdf1 = Query(theQuery, cPT)
    AppendList("Results01.txt", rootPdf1)
#   Query a naive Bayesian network end

    # Continue filling the results.txt file here ...