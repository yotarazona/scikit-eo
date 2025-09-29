<!-- markdownlint-disable -->

<a href="..\scikeo\deeplearning.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `deeplearning`





---

<a href="..\scikeo\deeplearning.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `processing`

```python
processing(
    raster_path,
    label_path=None,
    patch_size=256,
    export_patches=False,
    output_dir=None,
    export_labels=False,
    labels_output_dir=None,
    padding_mode='constant',
    padding_value=0,
    overlap=0
)
```

Reads raster and labels, splits them into patches with optional overlap and padding. 



**Args:**
 
 - <b>`raster_path`</b> (str):  Path to raster file 
 - <b>`label_path`</b> (str, optional):  Path to label raster 
 - <b>`patch_size`</b> (int):  Patch size 
 - <b>`export_patches`</b> (bool):  Whether to export image patches 
 - <b>`output_dir`</b> (str, optional):  Directory to export image patches 
 - <b>`export_labels`</b> (bool):  Whether to export label patches 
 - <b>`labels_output_dir`</b> (str, optional):  Directory to export label patches 
 - <b>`padding_mode`</b> (str):  'constant' or 'reflect' 
 - <b>`padding_value`</b>:  Value for constant padding 
 - <b>`overlap`</b> (int):  Number of pixels of overlap between patches 



**Returns:**
 
 - <b>`tuple`</b>:  (X_patches, y_patches) or X_patches if no labels are provided 


---

<a href="..\scikeo\deeplearning.py#L264"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `make_mean_iou_metric`

```python
make_mean_iou_metric(num_classes)
```

Create a stable Keras metric named 'mean_iou' to allow monitoring 'val_mean_iou'. 


---

<a href="..\scikeo\deeplearning.py#L275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `trainUnet`

```python
trainUnet(
    X_train,
    y_train,
    input_shape,
    num_classes,
    dropout_rate=0.2,
    learning_rate=0.0001,
    batch_size=16,
    epochs=50,
    validation_data=None,
    validation_split=0.0,
    data_augmentation=False,
    normalize=True,
    fill_nulls=True,
    null_value=0.0,
    save_best_model=True,
    model_path='best_model.keras'
)
```

Train a U-Net model for binary or multiclass semantic segmentation. 

Parameters 
---------- X_train : np.ndarray  Training feature patches of shape (N, H, W, C). y_train : np.ndarray  Training labels of shape (N, H, W, 1) for binary  or (N, H, W) with integer class labels for multiclass. input_shape : tuple  Shape of input tensors (H, W, C). num_classes : int  Number of output classes. Use 1 for binary segmentation. dropout_rate : float, optional  Dropout rate for regularization (default=0.2). learning_rate : float, optional  Learning rate for Adam optimizer (default=1e-4). batch_size : int, optional  Number of samples per gradient update (default=16). epochs : int, optional  Number of training epochs (default=50). validation_data : tuple, optional  Tuple (X_val, y_val) used for validation. If None,  a split from training data can be created with `validation_split`. validation_split : float, optional  Fraction of training data reserved for validation if  `validation_data` is not provided (default=0.0). data_augmentation : bool, optional  If True, applies random flips and rotations for augmentation  (default=False). normalize : bool, optional  If True, applies per-band normalization using StandardScaler (default=True). fill_nulls : bool, optional  If True, replaces NaN values with `null_value` (default=True). null_value : float, optional  Value used to replace NaN values (default=0.0). save_best_model : bool, optional  If True, saves the best model according to validation IoU (default=True). model_path : str, optional  File path to save the best model (default="best_model.keras"). 

Returns 
------- model : tensorflow.keras.Model  The trained U-Net model. history : keras.callbacks.History  Training history with loss and metric evolution. 


---

<a href="..\scikeo\deeplearning.py#L509"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `predictRaster`

```python
predictRaster(
    model,
    raster_path,
    patch_size=256,
    num_classes=1,
    output_path=None,
    fill_nulls=True,
    null_value=0,
    normalize=True,
    overlap=0
)
```

Predicts over a complete raster file and reconstructs the original image. 



**Args:**
 
 - <b>`model`</b>:  Trained model (U-Net) 
 - <b>`raster_path`</b> (str):  Path to the .tif file for prediction 
 - <b>`patch_size`</b> (int):  Patch size used during training 
 - <b>`num_classes`</b> (int):  Number of classes of the model 
 - <b>`output_path`</b> (str):  Path to save the predicted raster (optional) 
 - <b>`fill_nulls`</b> (bool):  Whether to replace null values with null_value 
 - <b>`null_value`</b>:  Value to replace nulls 
 - <b>`normalize`</b> (bool):  Whether to apply normalization 
 - <b>`overlap`</b> (int):  Number of overlapping pixels between patches 



**Returns:**
 
 - <b>`np.array`</b>:  Predicted reconstructed image 


---

<a href="..\scikeo\deeplearning.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `reconstruct_from_patches`

```python
reconstruct_from_patches(
    patches,
    original_height,
    original_width,
    patch_size,
    overlap=0
)
```

Reconstructs a full image from patches, averaging values in overlapped regions. 


---

<a href="..\scikeo\deeplearning.py#L632"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `save_prediction_raster`

```python
save_prediction_raster(
    prediction,
    output_path,
    original_profile,
    original_transform
)
```

Saves prediction as a georeferenced raster. 




---

_This file was automatically generated via [lazydocs](https://github.com/ml-tooling/lazydocs)._
