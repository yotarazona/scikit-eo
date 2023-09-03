<!-- markdownlint-disable -->

<a href="..\scikeo\deeplearning.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `deeplearning`






---

<a href="..\scikeo\deeplearning.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `DL`
Deep Learning classification in Remote Sensing 

<a href="..\scikeo\deeplearning.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(image, endmembers, nodata=-99999)
```

Parameter: 

 image: Optical images. It must be rasterio.io.DatasetReader with 3d.  

 endmembers: Endmembers must be a matrix (numpy.ndarray) and with more than one endmember.   Rows represent the endmembers and columns represent the spectral bands.  The number of bands must be equal to the number of endmembers.  E.g. an image with 6 bands, endmembers dimension should be $n*6$, where $n$   is rows with the number of endmembers and 6 is the number of bands   (should be equal).  In addition, Endmembers must have a field (type int or float) with the names   of classes to be predicted.  

 nodata: The NoData value to replace with -99999.  






---

<a href="..\scikeo\deeplearning.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FullyConnected`

```python
FullyConnected(
    hidden_layers=3,
    hidden_units=[64, 32, 16],
    output_units=10,
    input_shape=(6,),
    epochs=300,
    batch_size=32,
    training_split=0.8,
    random_state=None
)
```

This algorithm consiste of a network with a sequence of Dense layers, which area densely  connnected (also called *fully connected*) neural layers. This is the simplest of deep  learning. 



**Parameters:**
 


 - <b>`hidden_layers`</b>:  Number of hidden layers to be used. 3 is for default. 


 - <b>`hidden_units`</b>:  Number of units to be used. This is related to 'neurons' in each hidden   layers.  


 - <b>`output_units`</b>:  Number of clases to be obtained. 


 - <b>`input_shape`</b>:  The input shape is generally the shape of the input data provided to the   Keras model while training. The model cannot know the shape of the   training data. The shape of other tensors(layers) is computed automatically. 


 - <b>`epochs`</b>:  Number of iteration, the network will compute the gradients of the weights with  regard to the loss on the batch, and update the weights accordingly. 


 - <b>`batch_size`</b>:  This break the data into small batches. In deep learning, models do not   process antire dataset at once. 


 - <b>`training_split`</b>:  For splitting samples into two subsets, i.e. training data and for testing  data. 


 - <b>`random_state`</b>:  Random state ensures that the splits that you generate are reproducible.  
 - <b>`Please, see for more details https`</b>: //scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html 

Return: 

A dictionary with Labels of classification as numpy object, overall accuracy,  among others results. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
