import numpy as np
import gzip 

class ImportTrainData:

    def __init__(self,path):
        if path != None:
            self.f = gzip.open(path,'rb')         # file object created

            """ Structure of the training data file :
              Header:-
                1st 4 bytes  - magic number(32 bit integer)
                2nd 4 bytes  - number of images(32 bit integer)
                3rd 4 bytes  - number of rows(32 bit integer)
                4rt 4 bytes  - number of cols(32 bit integer)
              Data:-
                unsigned byte - pixel(0-255)
                unsigned byte - pixel(0-255)

              Magic Number:-
                First 2 bytes are always 0
                third byte codes the type of data
                  0x08 - unsigned byte
                  0x09 - signed byte
                  0x0B - short(2 bytes)
                  0x0C - int(4 bytes)
                  0x0D - float(4 bytes)
                  0x0E - double(8 bytes)
            """
        # the data type dictionary as defined for idx format files in MNIST
        self.dtypedict = {0x08:1,\
                          0x09:1,\
                          0x0B:2,\
                          0x0C:4,\
                          0x0D:4,\
                          0x0E:5}  

                              
         
    def getArray(self):
        """
        builds & returns a numpy array from the input file & its header

        """
        k = (self.f).read(4)                # 1st 4 bytes is magic number
        # Components of magic number
        j = memoryview(k[2]).tolist()       # tuple of the 3rd byte, ignore first 2 bytes as they are 0s
        self.dtype = self.dtypedict[j[0]]   # type of data in bytes,  this needs modification
        j = memoryview(k[3]).tolist()       # tuple of the 4rth byte
        self.ndim = j[0]                    # no of dimensions

        # build a sizeofdim array. No of dimensions of inp data is array size, array element defines size of the dimension
        self.sizeofdim = []
        for n in range(self.ndim):
            k = (self.f).read(4)            # size of a dim is a 4 byte int
            self.sizeofdim.append(sum([ memoryview(k[x]).tolist()[0]*(2**(8*(3-x))) for x in range(4) ]))
    
        """
        k = (self.f).read(4)                # 2nd 4 bytes - 32 bit int  - no of images(dimension 3)
        # big endian conversion to int
        self.noofimages = sum([ memoryview(k[x]).tolist()[0]*(2**(8*(3-x))) for x in range(4) ]) 

        k = (self.f).read(4)                # 3rd 4 bytes - no of rows(dimension 2)
        # big endian conversion to int
        self.rows = sum([ memoryview(k[x]).tolist()[0]*(2**(8*(3-x))) for x in range(4) ]) 

        k = (self.f).read(4)                # 4rth 4 bytes - no of cols(dimension 1)
        # big endian conversion to int
        self.cols = sum([ memoryview(k[x]).tolist()[0]*(2**(8*(3-x))) for x in range(4) ]) 
        
        # Training data is converted into a 2 dimensional array(60000 rows & 785 cols)
        # Every Training Input Image consists of 28*28 pixels, that is 784 pixels
        # Each Training Output label is one of the set(0-9)
        #  Thus the training data array consists of the input image info & the corresponding output
        
        # pixels = (self.f).read(self.noofimages*self.rows*self.cols)
        """
        pixels = (self.f).read(np.prod(self.sizeofdim))
        
        """
        self.train_data = np.array([ memoryview(pixels[x]).tolist()[0] \
                                     for x in range(self.noofimages*self.rows*self.cols)]).reshape(self.noofimages,self.rows*self.cols)
        """
        return np.array([ memoryview(pixels[x]).tolist()[0] \
                                     for x in range(np.prod(self.sizeofdim))]).reshape(tuple(self.sizeofdim))


                                      
        
                                      
                                      
        

        
        
        
            

    
