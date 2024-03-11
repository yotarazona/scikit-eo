<!-- markdownlint-disable -->

<a href="..\scikeo\mla.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `mla`






---

<a href="..\scikeo\mla.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `MLA`
Supervised classification in Remote Sensing 

<a href="..\scikeo\mla.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(image, endmembers, nodata=-99999)
```

Parameter: 

 image: Optical images. It must be rasterio.io.DatasetReader with 3d.  

 endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember.   Rows represent the endmembers and columns represent the spectral bands.  The number of bands must be equal to the number of endmembers.  E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$   is rows with the number of endmembers and 6 is the number of bands   (should be equal).  In addition, Endmembers must have a field (type int or float) with the names   of classes to be predicted.  

 nodata: The NoData value to replace with -99999.  






---

<a href="..\scikeo\mla.py#L238"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `DT`

```python
DT(training_split=0.8, random_state=None, **kwargs)
```

Decision Tree is also a supervised non-parametric statistical learning technique, where the input data is divided recursively  into branches depending on certain decision thresholds until the data are segmented into homogeneous subgroups.  This technique has substantial advantages for remote sensing classification problems due to its flexibility, intuitive simplicity,  and computational efficiency. 

DT support raster data read by rasterio (rasterio.io.DatasetReader) as input. 





**Parameters:**
 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data.  


 - <b>`**kwargs`</b>:  These will be passed to DT, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier 

Return: 

A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix. 

---

<a href="..\scikeo\mla.py#L467"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `NB`

```python
NB(training_split=0.8, random_state=None, **kwargs)
```

Naive Bayes classifier is an effective and simple method for image classification based on probability theory. The NB  classifier assumes an underlying probabilistic model and captures the uncertainty about the model in a principled way,  that is, by calculating the occurrence probabilities of different attribute values for different classes in a training  set. 

NB support raster data read by rasterio (rasterio.io.DatasetReader) as input. 





**Parameters:**
 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data.  


 - <b>`**kwargs`</b>:  These will be passed to SVM, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html 

Return: 

A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix. 

---

<a href="..\scikeo\mla.py#L582"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `NN`

```python
NN(training_split=0.8, max_iter=300, random_state=None, **kwargs)
```

This classification consists of a neural network that is organized into several layers, that is, an input layer of predictor  variables, one or more layers of hidden nodes, in which each node represents an activation function acting on a weighted input  of the previous layers’ outputs, and an output layer. 

NN support raster data read by rasterio (rasterio.io.DatasetReader) as input. 





**Parameters:**
 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data.  


 - <b>`**kwargs`</b>:  These will be passed to SVM, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 

Return: 

A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix. 

---

<a href="..\scikeo\mla.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `RF`

```python
RF(training_split=0.8, random_state=None, **kwargs)
```

Random Forest is a derivative of Decision Tree which provides an improvement over DT to overcome the weaknesses of a single DT.  The prediction model of the RF classifier only requires two parameters to be identified: the number of classification trees desired,  known as “ntree,” and the number of prediction variables, known as “mtry,” used in each node to make the tree grow. 

RF support raster data read by rasterio (rasterio.io.DatasetReader) as input. 





**Parameters:**
 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data.  


 - <b>`**kwargs`</b>:  These will be passed to RF, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html 

Return: 

A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix. 

---

<a href="..\scikeo\mla.py#L117"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SVM`

```python
SVM(training_split=0.8, random_state=None, kernel='linear', **kwargs)
```

The Support Vector Machine (SVM) classifier is a supervised non-parametric statistical learning technique that  does not assume a preliminary distribution of input data. Its discrimination criterion is a  hyperplane that separates the classes in the multidimensional space in which the samples  that have established the same classes are located, generally some training areas. 

SVM support raster data read by rasterio (rasterio.io.DatasetReader) as input. 





**Parameters:**
 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data. 


 - <b>`kernel `</b>:  {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}, default='rbf' Specifies   the kernel type to be used in the algorithm. It must be one of 'linear', 'poly',   'rbf', 'sigmoid', 'precomputed' or a callable. If None is given, 'rbf' will  
 - <b>`be used. See https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC for more details. 


 - <b>`**kwargs`</b>:  These will be passed to SVM, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC 

Return: 

A dictionary containing labels of classification as numpy object, overall accuracy, kappa index, confusion matrix. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
