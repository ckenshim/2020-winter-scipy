def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1]
    mean = np.mean(realData, axis = 0)

 	return array(mean)


a = np.arange(15).reshape(3, 5)
print(Mean(a))