# -*- coding: utf-8 -*-
# +
import copy
import rasterio
import numpy as np
import pandas as pd
from dbfread import DBF
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import warnings

class MLA(object):
    
    'Supervised classification in Remote Sensing'
    
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
            
        
    def SVM(self, training_split = 0.8, random_state = None, kernel = 'linear', **kwargs):
        
        '''The Support Vector Machine (SVM) classifier is a supervised non-parametric statistical learning technique that 
        does not assume a preliminary distribution of input data. Its discrimination criterion is a 
        hyperplane that separates the classes in the multidimensional space in which the samples 
        that have established the same classes are located, generally some training areas.
        
        SVM support raster data read by rasterio (rasterio.io.DatasetReader) as input.
     
        
        Parameters:
    
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
    
            kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf' Specifies 
                     the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 
                     'rbf', 'sigmoid', 'precomputed' or a callable. If None is given, 'rbf' will 
                     be used. See https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
                     for more details.
                     
            **kwargs: These will be passed to SVM, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    
        Return:
        
            A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
        
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X.values, 
            y.values, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        # applying a support vector machine
        inst_svm = SVC(kernel = kernel, **kwargs)
        
        # model trained
        mt_svm = inst_svm.fit(Xtrain, ytrain)
        
        labels_svm = mt_svm.predict(self.arr)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels_svm = labels_svm[self.posIndx]
            self.class_final[self.posIndx] = labels_svm
            classSVM = self.class_final.reshape((self.rows, self.cols))
        else:
            classSVM = labels_svm.reshape((self.rows, self.cols))
        
        # Confusion matrix
        predic_Xtest = mt_svm.predict(Xtest)
        
        MC = confusion_matrix(ytest, predic_Xtest)
    
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

        min_class = np.nanmin(labels_svm)
        max_class = np.nanmax(labels_svm)
        
        namesCol = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predic_Xtest)
        kappa = cohen_kappa_score(ytest, predic_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': classSVM,
                  'Image': self.image
                 } 
    
        return output
    
    def DT(self, training_split = 0.8, random_state = None, **kwargs):
        
        '''Decision Tree is also a supervised non-parametric statistical learning technique, where the input data is divided recursively 
        into branches depending on certain decision thresholds until the data are segmented into homogeneous subgroups. 
        This technique has substantial advantages for remote sensing classification problems due to its flexibility, intuitive simplicity, 
        and computational efficiency.
        
        DT support raster data read by rasterio (rasterio.io.DatasetReader) as input.
        
        
        Parameters:
    
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
                     
            **kwargs: These will be passed to DT, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    
        Return:
        
            A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
            
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X.values, 
            y.values, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        # applying a support vector machine
        inst_dt = DT(**kwargs)
        
        # model trained
        mt_dt = inst_dt.fit(Xtrain, ytrain)
        
        labels_dt = mt_dt.predict(self.arr)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels_dt = labels_dt[self.posIndx]
            self.class_final[self.posIndx] = labels_dt
            classDT = self.class_final.reshape((self.rows, self.cols))
        else:
            classDT = labels_dt.reshape((self.rows, self.cols))
        
        # Confusion matrix
        predic_Xtest = mt_dt.predict(Xtest)
        
        MC = confusion_matrix(ytest, predic_Xtest)
    
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

        min_class = np.nanmin(labels_dt)
        max_class = np.nanmax(labels_dt)
        
        namesCol = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predic_Xtest)
        kappa = cohen_kappa_score(ytest, predic_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': classDT,
                  'Image': self.image
                 } 
    
        return output
    
    def RF(self, training_split = 0.8, random_state = None, **kwargs):
        
        '''Random Forest is a derivative of Decision Tree which provides an improvement over DT to overcome the weaknesses of a single DT. 
        The prediction model of the RF classifier only requires two parameters to be identified: the number of classification trees desired, 
        known as “ntree,” and the number of prediction variables, known as “mtry,” used in each node to make the tree grow.
        
        RF support raster data read by rasterio (rasterio.io.DatasetReader) as input.
        
        
        Parameters:
    
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
                     
            **kwargs: These will be passed to RF, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
        Return:
        
            A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
            
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X.values, 
            y.values, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        # applying a support vector machine
        inst_rf = RF(**kwargs)
        
        # model trained
        mt_rf = inst_rf.fit(Xtrain, ytrain)
        
        labels_rf = mt_rf.predict(self.arr)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels_rf = labels_rf[self.posIndx]
            self.class_final[self.posIndx] = labels_rf
            classRF = self.class_final.reshape((self.rows, self.cols))
        else:
            classRF = labels_rf.reshape((self.rows, self.cols))
        
        # Confusion matrix
        predic_Xtest = mt_rf.predict(Xtest)
        
        MC = confusion_matrix(ytest, predic_Xtest)
    
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

        min_class = np.nanmin(labels_rf)
        max_class = np.nanmax(labels_rf)
        
        namesCol = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predic_Xtest)
        kappa = cohen_kappa_score(ytest, predic_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': classRF,
                  'Image': self.image
                 } 
    
        return output
    
    def NB(self, training_split = 0.8, random_state = None, **kwargs):
        
        '''Naive Bayes classifier is an effective and simple method for image classification based on probability theory. The NB 
        classifier assumes an underlying probabilistic model and captures the uncertainty about the model in a principled way, 
        that is, by calculating the occurrence probabilities of different attribute values for different classes in a training 
        set.
        
        NB support raster data read by rasterio (rasterio.io.DatasetReader) as input.
        
        
        Parameters:
    
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
                     
            **kwargs: These will be passed to SVM, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    
        Return:
        
            A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
            
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X.values, 
            y.values, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        # applying a support vector machine
        inst_nb = GNB(**kwargs)
        
        # model trained
        mt_nb = inst_nb.fit(Xtrain, ytrain)
        
        labels_nb = mt_nb.predict(self.arr)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels_nb = labels_nb[self.posIndx]
            self.class_final[self.posIndx] = labels_nb
            classNB = self.class_final.reshape((self.rows, self.cols))
        else:
            classNB = labels_nb.reshape((self.rows, self.cols))
        
        # Confusion matrix
        predic_Xtest = mt_nb.predict(Xtest)
        
        MC = confusion_matrix(ytest, predic_Xtest)
    
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

        min_class = np.nanmin(labels_nb)
        max_class = np.nanmax(labels_nb)
        
        namesCol = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predic_Xtest)
        kappa = cohen_kappa_score(ytest, predic_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': classNB,
                  'Image': self.image
                 } 
    
        return output
    
    def NN(self, training_split = 0.8, max_iter = 300, random_state = None, **kwargs):
        
        '''This classification consists of a neural network that is organized into several layers, that is, an input layer of predictor 
        variables, one or more layers of hidden nodes, in which each node represents an activation function acting on a weighted input 
        of the previous layers’ outputs, and an output layer.
        
        NN support raster data read by rasterio (rasterio.io.DatasetReader) as input.
        
        
        Parameters:
    
            training_split: For splitting samples into two subsets, i.e. training data and for testing
                            data.
                     
            **kwargs: These will be passed to SVM, please see full lists at:
                  https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
        Return:
        
            A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
            
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        # split in training and testing
        Xtrain, Xtest, ytrain, ytest = train_test_split(
            X.values, 
            y.values, 
            train_size = training_split, 
            test_size = 1 - training_split, 
            random_state = random_state)
        
        # applying neural network
        inst_nn = MLP(max_iter = max_iter, **kwargs)
        
        # model trained
        mt_nn = inst_nn.fit(Xtrain, ytrain)
        
        labels_nn = mt_nn.predict(self.arr)
        
        # dealing with nan
        if self.key_nan:
            # image border like nan
            labels_nn = labels_nn[self.posIndx]
            self.class_final[self.posIndx] = labels_nn
            classNN = self.class_final.reshape((self.rows, self.cols))
        else:
            classNN = labels_nn.reshape((self.rows, self.cols))
        
        # Confusion matrix
        predic_Xtest = mt_nn.predict(Xtest)
        
        MC = confusion_matrix(ytest, predic_Xtest)
    
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

        min_class = np.nanmin(labels_nn)
        max_class = np.nanmax(labels_nn)
        
        namesCol = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesCol.append(stri)

        namesCol.extend(['Total', 'Users_Accuracy', 'Commission'])

        namesRow = []
        for i in np.arange(min_class, max_class + 1):
            stri = str(i)
            namesRow.append(stri)

        namesRow.extend(['Total', 'Producer_Accuracy', 'Omission'])
        namesRow

        confusionMatrix = pd.DataFrame(cm, 
                                        columns = namesCol, 
                                        index = namesRow)

        oa = accuracy_score(ytest, predic_Xtest)
        kappa = cohen_kappa_score(ytest, predic_Xtest)
        
        output = {'Overall_Accuracy': oa,
                  'Kappa_Index': kappa,
                  'Confusion_Matrix': confusionMatrix,
                  'Classification_Map': classNN,
                  'Image': self.image
                 } 
    
        return output
