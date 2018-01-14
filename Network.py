import numpy as np
import math

""" network class defines the 3 layer neural network.
1st layer is the input layer(no weights, only biases)
2nd layer is the hidden layer(weights and biases)
3rd layer is the output layer(weights and biases)
neural network consists of the following attributes
1) The weight matrices for 2 and 3rd layer. For the nth layer, the matrix consists of j rows(neurons in nth layer)
                                                                               & k cols(neurons in (n-1)th layer)
2) The bias matrices for 1,2 and 3rd layers. For the nth layer, the matrix consists of j rows(neurons in nth layer) and 1 col
3) The input matrix consists of x1,x2,x3....xn,y

"""
# accepts a tuple of neurons in each layer
# for a three layered network of n1,n2,n3 neurons in each layer the tuple is (n1, n2, n3)
class Network:
    def __init__(self,netlayers):
        # list of weight matrices for each layer
        # no weights are defined for the input layer for obvious reasons
        self.weights = [ np.random.randn(netlayers[i+1],netlayers[i]) for i in range(netlayers.__len__()-1) ]

        # list of bias vectors for each layer
        # no biases are defined for the input layer, as the output is equal to the input
        self.biases = [ np.random.randn(netlayers[i+1],1) for i in range(netlayers.__len__()-1) ]
        self.netlayers = netlayers
        self.layers = len(netlayers)
        
        # mini batch size
        self.msize = 10;
        # no of epochs
        self.epoch = 30;

        # learning rate
        self.eta = 3.0;



    def loadInputs(self,train_data,test_data=None):
        # loads the inputs and starts the learning process
        # list of weighted inputs for each layer; weighted input for a neuron = sigma(weights*activationinputs) + bias of the neuron

        # Epoch - An complete set of mini batches equalling total training_data is an epoch
        # Mini batch - A random set of inputs with size = mbatchsize; 
        # An epoch is created by shuffling the training data set & splitting it into mini batches
        # Each learning step for all weights & biases is conducted over a full mini-batch. The process is repeated
        #       until all mini-batches in an epoch are exhausted. Then a new epoch is created & learning repeats 
        #       until epoch size is reached.
        # For each mini-batch:-
        #       For each input in the mbatch, delta is calculated for each layer.(back propagation)
        #                                     delc_bybias is calculated for each layer.
        #                                     delc_byweight is calculated for each layer.
        # Weights & biases are updated by one step.
        # Process is repeated until all mini-batches are done in the epoch
        # Epoch is updated & process repeats

        # Epoch creation start
        # shuffle the training_data. training_data is of form [(input1, label1),(input2, label2), (input3, label3)...]
        # create mini batches.
        for epoch in xrange(self.epoch):
            # shuffle the training data
            np.random.shuffle(train_data)
            
            mini_batches = [ train_data[i:i+self.msize] for i in xrange(0,len(train_data),self.msize)]
            for mini_batch in mini_batches:
                # For each mini-batch, delta(error should be calculated)
                # Back propagation algorithm
                part_dervw = [i-i for i in xrange(self.layers-1)]
                part_dervb = [i-i for i in xrange(self.layers-1)]
                for x in mini_batch:
                    z = []
                    a = []

                    # first layer activations are from input neurons
                    a.append(np.reshape(x[:-1],(1,len(x)-1)))
                    # compute activations & weighted inputs for each layer for the input in the mini-batch
                    for l in xrange(self.layers-1):
                        z.append(np.add(np.dot(a[l], np.transpose(self.weights[l])),np.transpose(self.biases[l])))
                        a.append(sigmoid(z[l]))
                        
                    y = np.zeros((self.netlayers[-1],1))
                    y[x[-1]] = 1.0
                    # delta_l(error) is not defined for input layer, hence self.nlayers-1
                    delta_l = [ i-i for i in xrange(self.layers-1) ]
                    # error for the last layer(output)
                    delta_l[self.layers-2] = np.multiply(a[-1]-np.transpose(y),sigmoid_derv(z[-1]))
                    # back propagate the error
                    for i in xrange(self.layers-2,0,-1):
                        delta_l[i-1] = np.transpose(np.multiply(np.dot(np.transpose(self.weights[i]),np.transpose(delta_l[i])),np.transpose(sigmoid_derv(z[i-1]))))

                    # partial derivativs of the weights
                    # dont forget to multiply by the eta
                    for i in xrange(self.layers-1):
                        part_dervw[i] = np.add(part_dervw[i],np.dot(np.transpose(delta_l[i]),a[i]))

                    # partial derivatives of the biases
                    # dont forget to multiply by the eta
                    for i in xrange(self.layers-1):
                        part_dervb[i] = np.add(part_dervb[i],np.transpose(delta_l[i]))

                # Update the weights & biases by a step after the minibatch is done
                for l in xrange(self.layers-1):
                    self.weights[l] = np.subtract(self.weights[l],(self.eta/self.msize)*part_dervw[l])
                    self.biases[l] = np.subtract(self.biases[l],(self.eta/self.msize)*part_dervb[l])

            # Test for this epoch
            print("Epoch no %d" % (epoch))
            self.test_network(test_data)
                
        print("Learning completed.....\n")            
    
    def test_network(self,test_data):
        # The test data is used for the validation of the learned weights & biases
        # The format of the test data remains the same as the training data
        tcount = 0
        for x in test_data:
            z = []
            a = []
            # first layer activations are from input neurons
            a.append(np.reshape(x[:-1],(1,len(x)-1)))
            # compute activations & weighted inputs for each layer for the input in the mini-batch
            for l in xrange(self.layers-1):
                z.append(np.add(np.dot(a[l], np.transpose(self.weights[l])),np.transpose(self.biases[l])))
                a.append(sigmoid(z[l]))
                
            # Calculating the no of correct activations from the network
            if x[-1] == a[self.layers-1].argmax():
                tcount += 1
        print("%d out of %d data points\n" % (tcount, len(test_data)))
   

def sigmoid(x):
        return 1.0/(1.0+np.exp(-x))

def sigmoid_derv(x):
        # The sigma-dash function used in
        return sigmoid(x)*(1-sigmoid(x))


        
        



        
