from loadinputs import ImportTrainData
import numpy as np

class Training_data:

    def __init__(self):
        k = ImportTrainData('/Users/grognard/Documents/Programs/AI/NN/train-images-idx3-ubyte.gz')
        self.train_data = k.getArray()

        k = ImportTrainData('/Users/grognard/Documents/Programs/AI/NN/train-labels-idx1-ubyte.gz')
        self.train_labels = k.getArray()
        self.train_data = self.train_data.reshape(60000,28*28)
        self.train_data = np.divide(self.train_data,255.0)
#        self.train_labels = np.divide(self.train_labels,1.0)

    def getData(self):
        # combining the training inputs & the labels into a single list of tuples
        self.full_data = []

        for i in xrange(60000):
            l = []
            l = list(self.train_data[i])
            l.append(self.train_labels[i])
            self.full_data.append(tuple(l))

        return self.full_data

        
