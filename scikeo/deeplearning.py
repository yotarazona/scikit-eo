# +
import copy
import rasterio
import numpy as np
import pandas as pd
from dbfread import DBF
from tensorflow import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

class DL(object):
    
    'Deep Learning classification in Remote Sensing'
    
    def __init__(self, image, endmembers, nodata = -99999):
        
        '''
        Parameter:
        
            image: Optical images. It must be rasterio.io.DatasetReader with 3d.
            
            endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember. 
                    Rows represent the endmembers and columns represent the spectral bands.
                    The number of bands must be equal to the number of endmembers.
                    E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$ 
                    is rows with the number of endmembers and 6 is the number of bands 
                    (should be equal).
                    In addition, Endmembers must have a field (type int or float) with the names 
                    of classes to be predicted.
            
            nodata: The NoData value to replace with -99999.
            
        '''
        
        self.image = image
        self.endm = endmembers
        self.nodata = nodata
        
        if not isinstance(self.image, (rasterio.io.DatasetReader)):
            raise TypeError('"image" must be raster read by rasterio.open().')
        
        bands = self.image.count
        
        rows = self.image.height
        
        cols = self.image.width
        
        st = image.read()
        
        # data in [rows, cols, bands]
        st_reorder = np.moveaxis(st, 0, -1) 
        # data in [rows*cols, bands]
        arr = st_reorder.reshape((rows*cols, bands))

        # nodata
        #if np.isnan(np.sum(arr)):
            #arr[np.isnan(arr)] = self.nodata
        
        # dealing with nan
        key_nan = np.isnan(np.sum(arr[:,0]))
        
        if key_nan:
            # saving un array for predicted classes
            class_final = arr[:, 0].copy()
            # positions with nan
            posIndx = np.argwhere(~np.isnan(class_final)).flatten()
            # replace np.nan -> 0
            arr[np.isnan(arr)] = 0
            
        # if it is read by pandas.read_csv()
        if isinstance(self.endm, (pd.core.frame.DataFrame)):
            
            for i in np.arange(self.endm.shape[1]):
                
                if all(self.endm.iloc[:,int(i)] < 100) & all(self.endm.iloc[:,int(i)] >= 0): indx = i; break
        
        # if the file is .dbf    
        elif isinstance(self.endm, (DBF)): # isinstance() function With Inheritance
            
            self.endm = pd.DataFrame(iter(self.endm))
            
            for i in np.arange(self.endm.shape[1]):
                
                if all(self.endm.iloc[:,int(i)] < 100) & all(self.endm.iloc[:,int(i)] >= 0): indx = i; break
        
        else:
            raise TypeError('"endm" must be .csv (pandas.core.frame.DataFrame).')

        if not self.endm.shape[1] == (bands + 1):
            raise ValueError('The number of columns of signatures (included the class column) must'
                             'be equal to the number of bands + 1.')
        
        self.indx = indx
        
        self.arr = arr
        
        self.rows = rows
        
        self.cols = cols
        
        if key_nan:
            self.key_nan = key_nan
            self.posIndx = posIndx
            self.class_final = class_final
        else:
            self.key_nan = key_nan
        
    def FullyConnected(self, hidden_layers = 3, hidden_units = [64, 32, 16], output_units = 10,
                       input_shape = (6,), epochs = 300, batch_size = 32, training_split = 0.8, 
                       random_state = None):
        
        ''' 
        This algorithm consists of a network with a sequence of Dense layers, which area densely 
        connnected (also called *fully connected*) neural layers. This is the simplest of deep 
        learning.
        
        Parameters:
    
            hidden_layers: Number of hidden layers to be used. 3 is for default.
    
            hidden_units: Number of units to be used. This is related to 'neurons' in each hidden 
                          layers.
                          
            output_units: Number of clases to be obtained.
            
            input_shape: The input shape is generally the shape of the input data provided to the 
                         Keras model while training. The model cannot know the shape of the 
                         training data. The shape of other tensors(layers) is computed automatically.
            
            epochs: Number of iteration, the network will compute the gradients of the weights with
                    regard to the loss on the batch, and update the weights accordingly.
            
            batch_size: This break the data into small batches. In deep learning, models do not 
                        process antire dataset at once.
            
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
            
            random_state: Random state ensures that the splits that you generate are reproducible. 
                          Please, see for more details https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
            
        Return:
        
            A dictionary with Labels of classification as numpy object, overall accuracy, 
            among others results.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
        
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X, 
            y, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        ytrain_catego = to_categorical(ytrain)
        ytest_catego = to_categorical(ytest)
        
        fcl = [layers.Dense(hidden_units[0], activation = 'relu', input_shape = input_shape)]
        
        for i in np.arange(1, hidden_layers):
            
            fcl.append(layers.Dense(hidden_units[i], activation = 'relu'))
        
        fcl.append(layers.Dense(output_units, activation = 'softmax'))
        
        model = models.Sequential(fcl)
        
        model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        # model trained
        model.fit(Xtrain, ytrain_catego, epochs = epochs, batch_size = batch_size, verbose = 0)
        
        labels_fullyconnected = model.predict(self.arr)
        
        labels = [np.argmax(element) for element in labels_fullyconnected]
        
        labels = np.array(labels)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels = labels[self.posIndx]
            self.class_final[self.posIndx] = labels
            class_fullyconnected = self.class_final.reshape((self.rows, self.cols))
        else:
            class_fullyconnected = labels.reshape((self.rows, self.cols))
  
        # Confusion matrix
        predict_prob = model.predict(Xtest)
        
        predict_Xtest = [np.argmax(element) for element in predict_prob]
        
        predict_Xtest = np.array(predict_Xtest)
        
        MC = confusion_matrix(ytest, predict_Xtest)
    
        MC = np.transpose(MC)

        # Users Accuracy and Commission
        total_rows = np.sum(MC, axis = 1)
        ua = np.diag(MC)/np.sum(MC, axis = 1)*100
        co = 100 - ua

        # Producer Accuracy and Comission
        total_cols = np.sum(MC, axis = 0)
        pa = np.diag(MC)/np.sum(MC, axis = 0)*100
        om = 100 - pa

        total_cols = np.concatenate([total_cols, np.repeat(np.nan, 3)])
        pa = np.concatenate([pa, np.repeat(np.nan, 3)])
        om = np.concatenate([om, np.repeat(np.nan, 3)])

        n = MC.shape[0]
        cm = np.concatenate([MC, total_rows.reshape((n,1)), ua.reshape((n,1)), 
                             co.reshape((n,1))], axis = 1)
        cm = np.concatenate([cm, total_cols.reshape((1,n+3)), pa.reshape((1,n+3)), 
                             om.reshape((1,n+3))])

        namesCol = []
        for i in np.arange(0, n):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(0, n):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predict_Xtest)
        kappa = cohen_kappa_score(ytest, predict_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': class_fullyconnected,
                  'Image': self.image
                 } 
    
        return output
    
# -


