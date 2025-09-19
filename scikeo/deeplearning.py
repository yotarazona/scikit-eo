# +
import numpy as np
import rasterio
from rasterio.windows import Window
import os

import numpy as np
import rasterio
from rasterio.windows import Window
import os

def process_raster_patches(raster_path, label_path=None, patch_size=256, 
                          normalize=False, export_patches=False, output_dir=None,
                          padding_mode='constant', padding_value=0):
    """
    Lee raster y etiquetas, divide en parches con padding si es necesario
    
    Args:
        raster_path (str): Ruta al archivo raster
        label_path (str, optional): Ruta al archivo de etiquetas
        patch_size (int): Tamaño de los parches
        normalize (bool): Si se debe normalizar los datos
        export_patches (bool): Si se deben exportar los parches
        output_dir (str, optional): Directorio para exportar parches
        padding_mode (str): 'constant' o 'reflect'
        padding_value: Valor para relleno constante
    
    Returns:
        tuple: (X_patches, y_patches) o X_patches si no hay etiquetas
    """
    # Leer raster
    with rasterio.open(raster_path) as src:
        raster_data = src.read()
        profile = src.profile
        height, width = src.height, src.width
        n_bands = src.count
    
    # Mover bandas a la última dimensión
    raster_data = np.moveaxis(raster_data, 0, -1)
    
    # Calcular número de parches necesarios (redondeando hacia arriba)
    n_patches_h = (height + patch_size - 1) // patch_size
    n_patches_w = (width + patch_size - 1) // patch_size
    total_patches = n_patches_h * n_patches_w
    
    # Aplicar padding si es necesario
    pad_h = (n_patches_h * patch_size - height) if height % patch_size != 0 else 0
    pad_w = (n_patches_w * patch_size - width) if width % patch_size != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        if padding_mode == 'reflect':
            raster_data = np.pad(raster_data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            raster_data = np.pad(raster_data, ((0, pad_h), (0, pad_w), (0, 0)), 
                                mode='constant', constant_values=padding_value)
    
    # Crear array para parches
    X_patches = np.zeros((total_patches, patch_size, patch_size, n_bands), dtype=raster_data.dtype)
    
    # Leer y aplicar padding a etiquetas si están disponibles
    if label_path is not None:
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
        
        if pad_h > 0 or pad_w > 0:
            if padding_mode == 'reflect':
                label_data = np.pad(label_data, ((0, pad_h), (0, pad_w)), mode='reflect')
            else:
                label_data = np.pad(label_data, ((0, pad_h), (0, pad_w)), 
                                   mode='constant', constant_values=padding_value)
        
        y_patches = np.zeros((total_patches, patch_size, patch_size), dtype=label_data.dtype)
    
    # Extraer parches
    patch_idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start = i * patch_size
            y_end = y_start + patch_size
            x_start = j * patch_size
            x_end = x_start + patch_size
            
            # Extraer parche
            patch = raster_data[y_start:y_end, x_start:x_end, :]
            X_patches[patch_idx] = patch
            
            # Extraer parche de etiquetas si está disponible
            if label_path is not None:
                label_patch = label_data[y_start:y_end, x_start:x_end]
                y_patches[patch_idx] = label_patch
            
            # Exportar parches si está habilitado
            if export_patches and output_dir is not None:
                _export_patch(patch, patch_idx, output_dir, profile, patch_size)
            
            patch_idx += 1
    
    if normalize:
        min_val = np.min(X_patches)
        max_val = np.max(X_patches)
        if max_val > min_val:
            X_patches = (X_patches - min_val) / (max_val - min_val)
    
    if label_path is not None:
        return X_patches, y_patches  # Solo devuelve 2 valores ahora
    else:
        return X_patches  # Solo devuelve 1 valor

def _export_patch(patch, patch_idx, output_dir, profile, patch_size):
    """Exporta un parche como archivo TIFF (función interna)"""
    # Reorganizar dimensiones para rasterio
    patch_export = np.moveaxis(patch, -1, 0)
    
    # Actualizar perfil
    patch_profile = profile.copy()
    patch_profile.update({
        'height': patch_size,
        'width': patch_size,
        'transform': rasterio.windows.transform(
            Window(0, 0, patch_size, patch_size), 
            patch_profile['transform']
        )
    })
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar parche
    output_path = os.path.join(output_dir, f'patch_{patch_idx:04d}.tif')
    with rasterio.open(output_path, 'w', **patch_profile) as dst:
        dst.write(patch_export.astype(patch_profile['dtype']))


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def train_unet(X_train, y_train, input_shape, num_classes, 
               dropout_rate=0.2, learning_rate=1e-4, batch_size=16, 
               epochs=50, validation_data=None, data_augmentation=False):
    """
    Entrena un modelo U-Net
    
    Args:
        X_train (np.array): Parches de entrenamiento
        y_train (np.array): Etiquetas de entrenamiento
        input_shape (tuple): Forma de entrada (alto, ancho, canales)
        num_classes (int): Número de clases
        dropout_rate (float): Tasa de dropout (default: 0.2)
        learning_rate (float): Tasa de aprendizaje (default: 1e-4)
        batch_size (int): Tamaño del lote (default: 16)
        epochs (int): Número de épocas (default: 50)
        validation_data (tuple): Datos de validación (X_val, y_val) (opcional)
        data_augmentation (bool): Si se debe usar aumento de datos (default: False)
    
    Returns:
        Model: Modelo entrenado
        History: Historial de entrenamiento
    """
    # Construir modelo U-Net
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(dropout_rate)(pool1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(dropout_rate)(pool2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(dropout_rate)(pool3)
    
    # Middle
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Decoder
    up5 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    up5 = concatenate([up5, conv3])
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
    conv5 = Dropout(dropout_rate)(conv5)
    
    up6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv2])
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
    conv6 = Dropout(dropout_rate)(conv6)
    
    up7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv1])
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
    conv7 = Dropout(dropout_rate)(conv7)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    outputs = Conv2D(num_classes, 1, activation=activation)(conv7)
    
    # Compilar modelo
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=loss,
                 metrics=['accuracy'])
    
    # Preparar etiquetas
    if num_classes > 1:
        y_train_cat = to_categorical(y_train, num_classes)
        if validation_data is not None:
            X_val, y_val = validation_data
            validation_data = (X_val, to_categorical(y_val, num_classes))
    else:
        y_train_cat = y_train
    
    # Entrenar modelo
    if data_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        )
        
        history = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
    else:
        history = model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
    
    return model, history

def train_deeplabv3(X_train, y_train, input_shape, num_classes, 
                   dropout_rate=0.2, learning_rate=1e-4, batch_size=16, 
                   epochs=50, validation_data=None, data_augmentation=False):
    """
    Entrena un modelo DeepLabV3 (versión simplificada)
    
    Args:
        X_train (np.array): Parches de entrenamiento
        y_train (np.array): Etiquetas de entrenamiento
        input_shape (tuple): Forma de entrada (alto, ancho, canales)
        num_classes (int): Número de clases
        dropout_rate (float): Tasa de dropout (default: 0.2)
        learning_rate (float): Tasa de aprendizaje (default: 1e-4)
        batch_size (int): Tamaño del lote (default: 16)
        epochs (int): Número de épocas (default: 50)
        validation_data (tuple): Datos de validación (X_val, y_val) (opcional)
        data_augmentation (bool): Si se debe usar aumento de datos (default: False)
    
    Returns:
        Model: Modelo entrenado
        History: Historial de entrenamiento
    """
    # Construir modelo DeepLabV3 simplificado
    inputs = Input(input_shape)
    
    # Encoder con atrous convolutions
    x = Conv2D(64, 3, dilation_rate=1, padding='same', activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(128, 3, dilation_rate=2, padding='same', activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(256, 3, dilation_rate=4, padding='same', activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # ASPP simplificado
    branch1 = Conv2D(256, 1, padding='same', activation='relu')(x)
    branch2 = Conv2D(256, 3, dilation_rate=6, padding='same', activation='relu')(x)
    branch3 = Conv2D(256, 3, dilation_rate=12, padding='same', activation='relu')(x)
    
    # Concatenar branches
    x = concatenate([branch1, branch2, branch3])
    x = Conv2D(256, 1, padding='same', activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Output
    if num_classes == 1:
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        activation = 'softmax'
        loss = 'categorical_crossentropy'
    
    outputs = Conv2D(num_classes, 1, activation=activation)(x)
    
    # Compilar modelo
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                 loss=loss,
                 metrics=['accuracy'])
    
    # Preparar etiquetas y entrenar (similar a train_unet)
    if num_classes > 1:
        y_train_cat = to_categorical(y_train, num_classes)
        if validation_data is not None:
            X_val, y_val = validation_data
            validation_data = (X_val, to_categorical(y_val, num_classes))
    else:
        y_train_cat = y_train
    
    # Entrenar modelo
    if data_augmentation:
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant'
        )
        
        history = model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
    else:
        history = model.fit(
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
    
    return model, history


from sklearn.metrics import confusion_matrix, accuracy_score

def predict_raster(model, raster_path, patch_size=256, output_path=None):
    """
    Realiza predicción sobre un raster completo
    
    Args:
        model: Modelo entrenado
        raster_path (str): Ruta al raster a predecir
        patch_size (int): Tamaño de los parches (default: 256)
        output_path (str, optional): Ruta para guardar la predicción
    
    Returns:
        np.array: Predicción completa
    """
    # Preprocesar imagen
    X_patches = process_raster_patches(raster_path, patch_size=patch_size, normalize=True)
    
    # Realizar predicción
    predictions = model.predict(X_patches)
    
    # Reconstruir imagen completa
    full_prediction = _reconstruct_image(predictions, raster_path, patch_size)
    
    # Guardar predicción si se especifica
    if output_path is not None:
        _save_prediction(full_prediction, raster_path, output_path)
    
    return full_prediction

def calculate_metrics(prediction, reference_path):
    """
    Calcula métricas de evaluación comparando con referencia
    
    Args:
        prediction (np.array): Predicción a evaluar
        reference_path (str): Ruta al raster de referencia
    
    Returns:
        dict: Métricas de evaluación
    """
    # Leer referencia
    with rasterio.open(reference_path) as src:
        reference = src.read(1)
    
    # Asegurar que las dimensiones coincidan
    if reference.shape != prediction.shape[:2]:
        from skimage.transform import resize
        prediction = resize(prediction, reference.shape, order=0, preserve_range=True)
    
    # Binarizar predicción si es necesario
    if len(prediction.shape) > 2 and prediction.shape[2] > 1:
        # Para múltiples clases, tomar la clase con mayor probabilidad
        prediction = np.argmax(prediction, axis=2)
    elif len(prediction.shape) == 2:
        # Para binario, umbralizar en 0.5
        prediction = (prediction > 0.5).astype(np.uint8)
    
    # Aplanar arrays para cálculo de métricas
    pred_flat = prediction.flatten()
    ref_flat = reference.flatten()
    
    # Calcular matriz de confusión
    cm = confusion_matrix(ref_flat, pred_flat)
    
    # Calcular métricas para cada clase
    metrics = {}
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        metrics[f'class_{i}'] = {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'dice': dice
        }
    
    # Calcular accuracy general
    metrics['overall_accuracy'] = accuracy_score(ref_flat, pred_flat)
    metrics['confusion_matrix'] = cm
    
    return metrics


def reconstruct_with_unpatchify(patches, original_shape, patch_size):
    """
    Reconstruye imagen usando enfoque unpatchify
    
    Args:
        patches: Array de parches de forma (n, patch_size, patch_size, c)
        original_shape: Forma original de la imagen (height, width)
        patch_size: Tamaño de los parches
    
    Returns:
        Imagen reconstruida
    """
    from patchify import unpatchify
    
    # Calcular la grilla de parches
    n_patches_h = (original_shape[0] + patch_size - 1) // patch_size
    n_patches_w = (original_shape[1] + patch_size - 1) // patch_size
    
    # Reorganizar parches en forma de grilla
    patches_grid = patches.reshape((n_patches_h, n_patches_w, patch_size, patch_size, -1))
    
    # Reconstruir imagen
    reconstructed = unpatchify(patches_grid, (n_patches_h * patch_size, n_patches_w * patch_size, patches.shape[-1]))
    
    # Recortar a tamaño original si se usó padding
    if reconstructed.shape[0] > original_shape[0] or reconstructed.shape[1] > original_shape[1]:
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]]
    
    return reconstructed
    

def _save_prediction(prediction, raster_path, output_path):
    """Guarda la predicción como raster (función interna)"""
    with rasterio.open(raster_path) as src:
        profile = src.profile
    
    # Actualizar perfil para la predicción
    profile.update(
        dtype=rasterio.float32,
        count=1 if len(prediction.shape) == 2 else prediction.shape[2],
        nodata=0
    )
    
    # Guardar predicción
    with rasterio.open(output_path, 'w', **profile) as dst:
        if len(prediction.shape) == 2:
            dst.write(prediction.astype(np.float32), 1)
        else:
            for i in range(prediction.shape[2]):
                dst.write(prediction[:, :, i].astype(np.float32), i + 1)

# -


