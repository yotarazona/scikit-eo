# -*- coding: utf-8 -*-
# +
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
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

class calmla(object):
    
    ''' **Calibrating supervised classification in Remote Sensing**
    
    This module allows to calibrate supervised classification in satellite images
    through various algorithms and using approaches such as Set-Approach, 
    Leave-One-Out Cross-Validation (LOOCV), Cross-Validation (k-fold) and 
    Monte Carlo Cross-Validation (MCCV)'''
    
    def __init__(self, endmembers):
        
        '''
        Parameter:
        
            endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember. 
                        Rows represent the endmembers and columns represent the spectral bands.
                        The number of bands must be equal to the number of endmembers.
                        E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$ 
                        is rows with the number of endmembers and 6 is the number of bands 
                        (should be equal).
                        In addition, Endmembers must have a field (type int or float) with the names 
                        of classes to be predicted.
        
        References:
        
            Tarazona, Y., Zabala, A., Pons, X., Broquetas, A., Nowosad, J., and Zurqani, H.A. 
            Fusing Landsat and SAR data for mapping tropical deforestation through machine learning 
            classification and the PVts-β non-seasonal detection approach, Canadian Journal of Remote 
            Sensing., vol. 47, no. 5, pp. 677–696, Sep. 2021.
      
        '''
        
        self.endm = endmembers
        
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
        
        self.indx = indx
        
        
    def splitData(self, random_state = None):
        
        ''' This method is to separate the dataset in predictor variables and the variable 
        to be predicted
        
        Parameter:
            
            self: Attributes of class calmla.
        
        Return:
            A dictionary with X and y.
        '''
        
        # removing the class column
        X = self.endm.drop(self.endm.columns[[self.indx]], axis = 1)
        
        # only predictor variables
        y = self.endm.iloc[:, self.indx]
        
        Xy_data = {'X': X, 'y': y}
        
        return Xy_data
    
    # Set-Approach
    
    def SA(self, split_data, models = ('svm', 'dt', 'rf'), train_size = 0.5, n_iter = 10, **kwargs):
        
        '''
        This module allows to calibrate supervised classification in satellite images 
        through various algorithms and using approaches such as Set-Approach.
        
        Parameters:
        
            split_data: A dictionary obtained from the *splitData* method of this package.
            
            models: Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'),
            Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter
            can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn').
            
            train_size: For splitting samples into two subsets, i.e. training data and 
            for testing data.
            
            n_iter: Number of iterations, i.e number of times the analysis is executed.
            
            **kwargs: These will be passed to SVM, DT, RF, NB and NN, please see full lists at:
                  https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
            
        Return:  
            A graphic with errors for each machine learning algorithms.
        '''
        svm_error_sa = []
        dt_error_sa = []
        rf_error_sa = []
        nb_error_sa = []
        nn_error_sa = []
        
        for i in range(n_iter):
            
            # split in training and testing
            Xtrain, Xtest, ytrain, ytest = train_test_split(split_data.get('X'), 
                                                            split_data.get('y'), 
                                                            train_size = train_size, 
                                                            test_size = 1 - train_size, 
                                                            random_state = None)
            if 'svm' in models:
                # applying a support vector machine
                inst_svm = SVC(**kwargs)
        
                # model trained
                mt_svm = inst_svm.fit(Xtrain, ytrain)
        
                # Confusion matrix
                predic_Xtest = mt_svm.predict(Xtest)
        
                oa = accuracy_score(ytest, predic_Xtest)
                svm_error_sa.append(1 - oa)
            
            if 'dt' in models:
                # applying a support vector machine
                inst_dt = DT(**kwargs)
        
                # model trained
                mt_dt = inst_dt.fit(Xtrain, ytrain)
        
                # Confusion matrix
                predic_Xtest = mt_dt.predict(Xtest)
        
                oa = accuracy_score(ytest, predic_Xtest)
                dt_error_sa.append(1 - oa)
            
            if 'rf' in models:
                # applying a support vector machine
                inst_rf = RF(**kwargs)
        
                # model trained
                mt_rf = inst_rf.fit(Xtrain, ytrain)
        
                # Confusion matrix
                predic_Xtest = mt_rf.predict(Xtest)
        
                oa = accuracy_score(ytest, predic_Xtest)
                rf_error_sa.append(1 - oa)
            
            if 'nb' in models:
                # applying a support vector machine
                inst_nb = GNB(**kwargs)
        
                # model trained
                mt_nb = inst_nb.fit(Xtrain, ytrain)
        
                # Confusion matrix
                predic_Xtest = mt_nb.predict(Xtest)
        
                oa = accuracy_score(ytest, predic_Xtest)
                nb_error_sa.append(1 - oa)
            
            if 'nn' in models:
                # applying a support vector machine
                inst_nn = MLP(max_iter = 400, **kwargs)
        
                # model trained
                mt_nn = inst_nn.fit(Xtrain, ytrain)
        
                # Confusion matrix
                predic_Xtest = mt_nn.predict(Xtest)
        
                oa = accuracy_score(ytest, predic_Xtest)
                nn_error_sa.append(1 - oa)
            
            
            dic = {'svm': svm_error_sa,
                  'dt': dt_error_sa,
                  'rf': rf_error_sa,
                  'nb': nb_error_sa,
                  'nn': nn_error_sa}

            all_models = ('svm', 'dt', 'rf', 'nb', 'nn')

            for model in all_models:
                if model not in models:
                    del dic[model]
                    
        errors_setApproach = dic
        
        return errors_setApproach 
        
        
    # Leave One Out Cross-Validation calibration
    
    def LOOCV(self, split_data, models = ('svm', 'dt'), cv = LeaveOneOut(), n_iter = 10, **kwargs):
        
        '''
        This module allows to calibrate supervised classification in satellite images 
        through various algorithms and using Leave One Out Cross-Validation.
        
        Parameters:
        
            split_data: A dictionary obtained from the *splitData* method of this package.
            
            models: Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'),
            Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter
            can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn').
            
            cv: For splitting samples into two subsets, i.e. training data and 
            for testing data. Following Leave One Out Cross-Validation.
            
            n_iter: Number of iterations, i.e number of times the analysis is executed.
            
            **kwargs: These will be passed to SVM, DT, RF, NB and NN, please see full lists at:
                  https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
            
        Return:  
            A graphic with errors for each machine learning algorithms. 
        '''
        svm_error_loocv = []
        dt_error_loocv = []
        rf_error_loocv = []
        nb_error_loocv = []
        nn_error_loocv = []
        
        #cv = LeaveOneOut()
        
        X = split_data.get('X')
        
        y = split_data.get('y')
        
        for i in range(n_iter):
            
            if 'svm' in models:
                # applying a support vector machine
                inst_svm = SVC(**kwargs)
                
                scores = cross_val_score(inst_svm, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                svm_error_loocv.append(1 - mean_scores)
            
            if 'dt' in models:
                # applying a support vector machine
                inst_dt = DT(**kwargs)
                
                scores = cross_val_score(inst_dt, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                dt_error_loocv.append(1 - mean_scores)
            
            if 'rf' in models:
                # applying a support vector machine
                inst_rf = RF(**kwargs)
                
                scores = cross_val_score(inst_rf, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                rf_error_loocv.append(1 - mean_scores)
            
            if 'nb' in models:
                # applying a support vector machine
                inst_nb = GNB(**kwargs)
                
                scores = cross_val_score(inst_nb, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                nb_error_loocv.append(1 - mean_scores)
            
            if 'nn' in models:
                # applying a support vector machine
                inst_nn = MLP(max_iter = 400, **kwargs)
                
                scores = cross_val_score(inst_nn, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                nn_error_loocv.append(1 - mean_scores)
            
            
            dic = {'svm': svm_error_loocv,
                  'dt': dt_error_loocv,
                  'rf': rf_error_loocv,
                  'nb': nb_error_loocv,
                  'nn': nn_error_loocv}

            all_models = ('svm', 'dt', 'rf', 'nb', 'nn')

            for model in all_models:
                if model not in models:
                    del dic[model]
                    
        errors_LOOCV = dic
        
        return errors_LOOCV
    
    # Cross-Validation Calibration
    
    def CV(self, split_data, models = ('svm', 'dt'), k = 5, n_iter = 10, random_state = None, **kwargs):
        
        '''
        This module allows to calibrate supervised classification in satellite images 
        through various algorithms and using Cross-Validation.
        
        Parameters:
        
            split_data: A dictionary obtained from the *splitData* method of this package.
            
            models: Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'),
            Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter
            can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn').
            
            cv: For splitting samples into two subsets, i.e. training data and 
            for testing data. Following Leave One Out Cross-Validation.
            
            n_iter: Number of iterations, i.e number of times the analysis is executed.
            
            **kwargs: These will be passed to SVM, DT, RF, NB and NN, please see full lists at:
                  https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
            
        Return:  
            A graphic with errors for each machine learning algorithms.
        '''
        
        svm_error_cv = []
        dt_error_cv = []
        rf_error_cv = []
        nb_error_cv = []
        nn_error_cv = []
        
        cv = KFold(n_splits = k, shuffle = True, random_state = random_state)
        
        #cv = LeaveOneOut()
        
        X = split_data.get('X')
        
        y = split_data.get('y')
        
        for i in range(n_iter):
            
            if 'svm' in models:
                # applying a support vector machine
                inst_svm = SVC(**kwargs)
                
                scores = cross_val_score(inst_svm, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                svm_error_cv.append(1 - mean_scores)
            
            if 'dt' in models:
                # applying a support vector machine
                inst_dt = DT(**kwargs)
                
                scores = cross_val_score(inst_dt, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                dt_error_cv.append(1 - mean_scores)
            
            if 'rf' in models:
                # applying a support vector machine
                inst_rf = RF(**kwargs)
                
                scores = cross_val_score(inst_rf, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                rf_error_cv.append(1 - mean_scores)
            
            if 'nb' in models:
                # applying a support vector machine
                inst_nb = GNB(**kwargs)
                
                scores = cross_val_score(inst_nb, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                nb_error_cv.append(1 - mean_scores)
            
            if 'nn' in models:
                # applying a support vector machine
                inst_nn = MLP(max_iter = 400, **kwargs)
                
                scores = cross_val_score(inst_nn, X, y, scoring = 'accuracy', cv = cv, n_jobs = -1)
                mean_scores = np.mean(scores)
                nn_error_cv.append(1 - mean_scores)
            
            
            dic = {'svm': svm_error_cv,
                  'dt': dt_error_cv,
                  'rf': rf_error_cv,
                  'nb': nb_error_cv,
                  'nn': nn_error_cv}

            all_models = ('svm', 'dt', 'rf', 'nb', 'nn')

            for model in all_models:
                if model not in models:
                    del dic[model]
                    
        errors_CV = dic
        
        return errors_CV
    
    # Monte Carlo Cross-Validation Calibration
    
    def MCCV(self, split_data, models = ('svm'), train_size = 0.5, n_splits = 5, n_iter = 10, random_state = None, **kwargs):
        
        '''
        This module allows to calibrate supervised classification in satellite images 
        through various algorithms and using Cross-Validation.
        
        Parameters:
        
            split_data: A dictionary obtained from the *splitData* method of this package.
            
            models: Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'),
            Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter
            can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn').
            
            cv: For splitting samples into two subsets, i.e. training data and 
            for testing data. Following Leave One Out Cross-Validation.
            
            n_iter: Number of iterations, i.e number of times the analysis is executed.
            
            **kwargs: These will be passed to SVM, DT, RF, NB and NN, please see full lists at:
                  https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
            
        Return:  
            A graphic with errors for each machine learning algorithms. 
        '''
        
        svm_error_mccv = []
        dt_error_mccv = []
        rf_error_mccv = []
        nb_error_mccv = []
        nn_error_mccv = []
        
        test_size = 1 - train_size
        
        mc = ShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = random_state)
        
        X = split_data.get('X')
        
        y = split_data.get('y')
        
        for i in range(n_iter):
            
            if 'svm' in models:
                # applying a support vector machine
                inst_svm = SVC(**kwargs)
                
                scores = cross_val_score(inst_svm, X, y, scoring = 'accuracy', cv = mc, n_jobs = -1)
                mean_scores = np.mean(scores)
                svm_error_mccv.append(1 - mean_scores)
            
            if 'dt' in models:
                # applying a support vector machine
                inst_dt = DT(**kwargs)
                
                scores = cross_val_score(inst_dt, X, y, scoring = 'accuracy', cv = mc, n_jobs = -1)
                mean_scores = np.mean(scores)
                dt_error_mccv.append(1 - mean_scores)
            
            if 'rf' in models:
                # applying a support vector machine
                inst_rf = RF(**kwargs)
                
                scores = cross_val_score(inst_rf, X, y, scoring = 'accuracy', cv = mc, n_jobs = -1)
                mean_scores = np.mean(scores)
                rf_error_mccv.append(1 - mean_scores)
            
            if 'nb' in models:
                # applying a support vector machine
                inst_nb = GNB(**kwargs)
                
                scores = cross_val_score(inst_nb, X, y, scoring = 'accuracy', cv = mc, n_jobs = -1)
                mean_scores = np.mean(scores)
                nb_error_mccv.append(1 - mean_scores)
            
            if 'nn' in models:
                # applying a support vector machine
                inst_nn = MLP(max_iter = 400, **kwargs)
                
                scores = cross_val_score(inst_nn, X, y, scoring = 'accuracy', cv = mc, n_jobs = -1)
                mean_scores = np.mean(scores)
                nn_error_mccv.append(1 - mean_scores)
            
            
            dic = {'svm': svm_error_mccv,
                  'dt': dt_error_mccv,
                  'rf': rf_error_mccv,
                  'nb': nb_error_mccv,
                  'nn': nn_error_mccv}

            all_models = ('svm', 'dt', 'rf', 'nb', 'nn')

            for model in all_models:
                if model not in models:
                    del dic[model]
                    
        errors_MCCV = dic
        
        return errors_MCCV

