<!-- markdownlint-disable -->

<a href="..\scikeo\calmla.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `calmla`






---

<a href="..\scikeo\calmla.py#L18"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `calmla`
**Calibrating supervised classification in Remote Sensing** 

This module allows to calibrate supervised classification in satellite images through various algorithms and using approaches such as Set-Approach,  Leave-One-Out Cross-Validation (LOOCV), Cross-Validation (k-fold) and  Monte Carlo Cross-Validation (MCCV) 

<a href="..\scikeo\calmla.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(endmembers)
```

Parameter: 

 endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember.   Rows represent the endmembers and columns represent the spectral bands.  The number of bands must be equal to the number of endmembers.  E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$   is rows with the number of endmembers and 6 is the number of bands   (should be equal).  In addition, Endmembers must have a field (type int or float) with the names   of classes to be predicted. 

References: 

 Tarazona, Y., Zabala, A., Pons, X., Broquetas, A., Nowosad, J., and Zurqani, H.A.   Fusing Landsat and SAR data for mapping tropical deforestation through machine learning   classification and the PVts-β non-seasonal detection approach, Canadian Journal of Remote   Sensing., vol. 47, no. 5, pp. 677–696, Sep. 2021. 




---

<a href="..\scikeo\calmla.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `CV`

```python
CV(
    split_data,
    models=('svm', 'dt'),
    k=5,
    n_iter=10,
    random_state=None,
    **kwargs
)
```

This module allows to calibrate supervised classification in satellite images  through various algorithms and using Cross-Validation. 



**Parameters:**
 


 - <b>`split_data`</b>:  A dictionary obtained from the *splitData* method of this package. 


 - <b>`models`</b>:  Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'), Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn'). 


 - <b>`cv`</b>:  For splitting samples into two subsets, i.e. training data and  for testing data. Following Leave One Out Cross-Validation. 


 - <b>`n_iter`</b>:  Number of iterations, i.e number of times the analysis is executed. 


 - <b>`**kwargs`</b>:  These will be passed to SVM, DT, RF, NB and NN, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/supervised_learning.html#supervised-learning 

Return:   A graphic with errors for each machine learning algorithms. 

---

<a href="..\scikeo\calmla.py#L223"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `LOOCV`

```python
LOOCV(split_data, models=('svm', 'dt'), cv=LeaveOneOut(), n_iter=10, **kwargs)
```

This module allows to calibrate supervised classification in satellite images  through various algorithms and using Leave One Out Cross-Validation. 



**Parameters:**
 


 - <b>`split_data`</b>:  A dictionary obtained from the *splitData* method of this package. 


 - <b>`models`</b>:  Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'), Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn'). 


 - <b>`cv`</b>:  For splitting samples into two subsets, i.e. training data and  for testing data. Following Leave One Out Cross-Validation. 


 - <b>`n_iter`</b>:  Number of iterations, i.e number of times the analysis is executed. 


 - <b>`**kwargs`</b>:  These will be passed to SVM, DT, RF, NB and NN, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/supervised_learning.html#supervised-learning 

Return:   A graphic with errors for each machine learning algorithms.  

---

<a href="..\scikeo\calmla.py#L422"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `MCCV`

```python
MCCV(
    split_data,
    models='svm',
    train_size=0.5,
    n_splits=5,
    n_iter=10,
    random_state=None,
    **kwargs
)
```

This module allows to calibrate supervised classification in satellite images  through various algorithms and using Cross-Validation. 



**Parameters:**
 


 - <b>`split_data`</b>:  A dictionary obtained from the *splitData* method of this package. 


 - <b>`models`</b>:  Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'), Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn'). 


 - <b>`cv`</b>:  For splitting samples into two subsets, i.e. training data and  for testing data. Following Leave One Out Cross-Validation. 


 - <b>`n_iter`</b>:  Number of iterations, i.e number of times the analysis is executed. 


 - <b>`**kwargs`</b>:  These will be passed to SVM, DT, RF, NB and NN, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/supervised_learning.html#supervised-learning 

Return:   A graphic with errors for each machine learning algorithms.  

---

<a href="..\scikeo\calmla.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SA`

```python
SA(split_data, models=('svm', 'dt', 'rf'), train_size=0.5, n_iter=10, **kwargs)
```

This module allows to calibrate supervised classification in satellite images  through various algorithms and using approaches such as Set-Approach. 



**Parameters:**
 


 - <b>`split_data`</b>:  A dictionary obtained from the *splitData* method of this package. 


 - <b>`models`</b>:  Models to be used such as Support Vector Machine ('svm'), Decision Tree ('dt'), Random Forest ('rf'), Naive Bayes ('nb') and Neural Networks ('nn'). This parameter can be passed like models = ('svm', 'dt', 'rf', 'nb', 'nn'). 


 - <b>`train_size`</b>:  For splitting samples into two subsets, i.e. training data and  for testing data. 


 - <b>`n_iter`</b>:  Number of iterations, i.e number of times the analysis is executed. 


 - <b>`**kwargs`</b>:  These will be passed to SVM, DT, RF, NB and NN, please see full lists at: 
 - <b>`https`</b>: //scikit-learn.org/stable/supervised_learning.html#supervised-learning 

Return:   A graphic with errors for each machine learning algorithms. 

---

<a href="..\scikeo\calmla.py#L74"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `splitData`

```python
splitData(random_state=None)
```

This method is to separate the dataset in predictor variables and the variable  to be predicted 

Parameter:  

 self: Attributes of class calmla. 

Return:  A dictionary with X and y. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
