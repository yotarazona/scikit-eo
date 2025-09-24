# +
# ==========================================
# deep learning module for scikit-eo
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dropout,Conv2DTranspose, concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint


# ==========================================
# PROCESSING FUNCTIONS
# ==========================================
def processing(raster_path, label_path, patch_size=256,
               export_patches=False, output_dir=None,
               export_labels=False, labels_output_dir=None,
               padding_mode='constant', padding_value=0,
               overlap=0):
    """
    Reads raster and labels, splits them into patches with optional overlap and padding.

    Args:
        raster_path (str): Path to raster file
        label_path (str, optional): Path to label raster
        patch_size (int): Patch size
        export_patches (bool): Whether to export image patches
        output_dir (str, optional): Directory to export image patches
        export_labels (bool): Whether to export label patches
        labels_output_dir (str, optional): Directory to export label patches
        padding_mode (str): 'constant' or 'reflect'
        padding_value: Value for constant padding
        overlap (int): Number of pixels of overlap between patches

    Returns:
        tuple: (X_patches, y_patches) or X_patches if no labels are provided
    """
    # Read raster
    with rasterio.open(raster_path) as src:
        raster_data = src.read()
        profile = src.profile
        height, width = src.height, src.width
        n_bands = src.count
        transform = src.transform

    # Move bands to last axis
    raster_data = np.moveaxis(raster_data, 0, -1)  # (H, W, C)

    # Step size considering overlap
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("❌ Overlap must be smaller than patch_size.")

    # Compute number of patches
    n_patches_h = (height - overlap + step - 1) // step
    n_patches_w = (width - overlap + step - 1) // step
    total_patches = n_patches_h * n_patches_w

    # Compute padding if needed
    pad_h = max(0, (n_patches_h - 1) * step + patch_size - height)
    pad_w = max(0, (n_patches_w - 1) * step + patch_size - width)

    if pad_h > 0 or pad_w > 0:
        if padding_mode == 'reflect':
            raster_data = np.pad(raster_data, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            raster_data = np.pad(raster_data, ((0, pad_h), (0, pad_w), (0, 0)),
                                 mode='constant', constant_values=padding_value)

    # Create image patch array
    X_patches = np.zeros((total_patches, patch_size, patch_size, n_bands), dtype=raster_data.dtype)

    # Read labels if available
    if label_path is not None:
        with rasterio.open(label_path) as src:
            label_data = src.read(1)
            label_profile = src.profile
            label_transform = src.transform

        if pad_h > 0 or pad_w > 0:
            if padding_mode == 'reflect':
                label_data = np.pad(label_data, ((0, pad_h), (0, pad_w)), mode='reflect')
            else:
                label_data = np.pad(label_data, ((0, pad_h), (0, pad_w)),
                                    mode='constant', constant_values=padding_value)

        y_patches = np.zeros((total_patches, patch_size, patch_size, 1), dtype=label_data.dtype)

    # Extract patches
    patch_idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start, x_start = i * step, j * step
            y_end, x_end = y_start + patch_size, x_start + patch_size

            # Image patch
            patch = raster_data[y_start:y_end, x_start:x_end, :]
            X_patches[patch_idx] = patch

            # Label patch
            if label_path is not None:
                label_patch = label_data[y_start:y_end, x_start:x_end]
                y_patches[patch_idx, :, :, 0] = label_patch

            # Export image patch
            if export_patches and output_dir is not None:
                patch_transform = rasterio.windows.transform(
                    Window(x_start, y_start, patch_size, patch_size),
                    transform
                )
                _export_patch(patch, patch_idx, output_dir, profile, patch_size, patch_transform)

            # Export label patch
            if label_path is not None and export_labels and labels_output_dir is not None:
                label_patch_transform = rasterio.windows.transform(
                    Window(x_start, y_start, patch_size, patch_size),
                    label_transform
                )
                _export_label_patch(label_patch, patch_idx, labels_output_dir,
                                    label_profile, patch_size, label_patch_transform)

            patch_idx += 1

    return (X_patches, y_patches) if label_path is not None else X_patches


def _export_patch(patch, patch_idx, output_dir, profile, patch_size, patch_transform):
    """Exports an image patch as GeoTIFF"""
    patch_export = np.moveaxis(patch, -1, 0)

    patch_profile = profile.copy()
    patch_profile.update({
        'height': patch_size,
        'width': patch_size,
        'transform': patch_transform
    })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'patch_{patch_idx:04d}.tif')

    with rasterio.open(output_path, 'w', **patch_profile) as dst:
        dst.write(patch_export.astype(patch_profile['dtype']))


def _export_label_patch(label_patch, patch_idx, output_dir, profile, patch_size, patch_transform):
    """Exports a label patch as GeoTIFF"""
    label_profile = profile.copy()
    label_profile.update({
        'height': patch_size,
        'width': patch_size,
        'count': 1,
        'dtype': label_patch.dtype,
        'transform': patch_transform
    })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'label_patch_{patch_idx:04d}.tif')

    with rasterio.open(output_path, 'w', **label_profile) as dst:
        dst.write(label_patch.astype(label_profile['dtype']), 1)


# ==========================================
# TRAINING FUNCTIONS
# ==========================================
def _mean_iou_core(y_true, y_pred, num_classes=1, smooth=1e-6):
    """Computes Mean IoU (binary or multiclass)."""
    if num_classes == 1:
        y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
        y_true_bin = tf.clip_by_value(y_true, 0.0, 1.0)

        intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1, 2, 3])
        union = (tf.reduce_sum(y_true_bin, axis=[1, 2, 3]) +
                 tf.reduce_sum(y_pred_bin, axis=[1, 2, 3]) - intersection)
        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)

    # Multiclass
    y_true_labels = tf.argmax(y_true, axis=-1)
    y_pred_labels = tf.argmax(y_pred, axis=-1)

    ious, weights = [], []
    for c in range(num_classes):
        y_true_c = tf.cast(tf.equal(y_true_labels, c), tf.float32)
        y_pred_c = tf.cast(tf.equal(y_pred_labels, c), tf.float32)

        intersection = tf.reduce_sum(y_true_c * y_pred_c, axis=[1, 2])
        union = (tf.reduce_sum(y_true_c, axis=[1, 2]) +
                 tf.reduce_sum(y_pred_c, axis=[1, 2]) - intersection)

        iou_c = (intersection + smooth) / (union + smooth)
        w_c = tf.cast(union > 0.0, tf.float32)

        ious.append(iou_c)
        weights.append(w_c)

    ious = tf.stack(ious, axis=0)
    weights = tf.stack(weights, axis=0)

    mean_iou_per_batch = tf.reduce_sum(ious * weights, axis=0) / (tf.reduce_sum(weights, axis=0) + smooth)
    return tf.reduce_mean(mean_iou_per_batch)

# ==========================================
# U-net 
# ==========================================
# ==============================
# MEAN IoU METRIC (binary and multiclass)
# ==============================
def _mean_iou_core(y_true, y_pred, num_classes=1, smooth=1e-6):

    """
    Compute Mean IoU for binary and multiclass cases.
    - Binary: threshold 0.5 on y_pred.
    - Multiclass: argmax across class channel.
    - Averages ignoring absent classes in the batch (union == 0).
    """
    if num_classes == 1:
        # y_true expected in {0,1}; y_pred in [0,1]
        y_pred_bin = tf.cast(y_pred >= 0.5, tf.float32)
        y_true_bin = tf.clip_by_value(y_true, 0.0, 1.0)

        intersection = tf.reduce_sum(y_true_bin * y_pred_bin, axis=[1, 2, 3])
        union = (tf.reduce_sum(y_true_bin, axis=[1, 2, 3]) +
                 tf.reduce_sum(y_pred_bin, axis=[1, 2, 3]) - intersection)
        iou = (intersection + smooth) / (union + smooth)
        return tf.reduce_mean(iou)

    # Multiclass
    y_true_labels = tf.argmax(y_true, axis=-1)  # (B,H,W)
    y_pred_labels = tf.argmax(y_pred, axis=-1)  # (B,H,W)

    ious = []
    weights = []
    for c in range(num_classes):
        y_true_c = tf.cast(tf.equal(y_true_labels, c), tf.float32)  # (B,H,W)
        y_pred_c = tf.cast(tf.equal(y_pred_labels, c), tf.float32)  # (B,H,W)

        intersection = tf.reduce_sum(y_true_c * y_pred_c, axis=[1, 2])
        union = (tf.reduce_sum(y_true_c, axis=[1, 2]) +
                 tf.reduce_sum(y_pred_c, axis=[1, 2]) - intersection)

        iou_c = (intersection + smooth) / (union + smooth)
        w_c = tf.cast(union > 0.0, tf.float32)  # ignore absent classes in the batch

        ious.append(iou_c)    # (B,)
        weights.append(w_c)   # (B,)

    ious = tf.stack(ious, axis=0)       # (C,B)
    weights = tf.stack(weights, axis=0) # (C,B)

    # Mean per batch, weighted by class presence
    mean_iou_per_batch = tf.reduce_sum(ious * weights, axis=0) / (tf.reduce_sum(weights, axis=0) + smooth)
    return tf.reduce_mean(mean_iou_per_batch)

def make_mean_iou_metric(num_classes):
    """Create a stable Keras metric named 'mean_iou' to allow monitoring 'val_mean_iou'."""
    def mean_iou_metric(y_true, y_pred):
        return _mean_iou_core(y_true, y_pred, num_classes=num_classes)
    mean_iou_metric.__name__ = "mean_iou"  # required so monitor is 'val_mean_iou'
    return mean_iou_metric


# ==============================
# UNET TRAINING (binary and multiclass)
# ==============================
def train_unet(X_train, y_train, input_shape, num_classes,
               dropout_rate=0.2, learning_rate=1e-4, batch_size=16,
               epochs=50, validation_data=None, validation_split=0.0,
               data_augmentation=False, normalize=True,
               fill_nulls=True, null_value=0.0,
               save_best_model=True, model_path="best_model.keras"):
    """
    Train a U-Net model for binary or multiclass semantic segmentation.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature patches of shape (N, H, W, C).
    y_train : np.ndarray
        Training labels of shape (N, H, W, 1) for binary
        or (N, H, W) with integer class labels for multiclass.
    input_shape : tuple
        Shape of input tensors (H, W, C).
    num_classes : int
        Number of output classes. Use 1 for binary segmentation.
    dropout_rate : float, optional
        Dropout rate for regularization (default=0.2).
    learning_rate : float, optional
        Learning rate for Adam optimizer (default=1e-4).
    batch_size : int, optional
        Number of samples per gradient update (default=16).
    epochs : int, optional
        Number of training epochs (default=50).
    validation_data : tuple, optional
        Tuple (X_val, y_val) used for validation. If None,
        a split from training data can be created with `validation_split`.
    validation_split : float, optional
        Fraction of training data reserved for validation if
        `validation_data` is not provided (default=0.0).
    data_augmentation : bool, optional
        If True, applies random flips and rotations for augmentation
        (default=False).
    normalize : bool, optional
        If True, applies per-band normalization using StandardScaler (default=True).
    fill_nulls : bool, optional
        If True, replaces NaN values with `null_value` (default=True).
    null_value : float, optional
        Value used to replace NaN values (default=0.0).
    save_best_model : bool, optional
        If True, saves the best model according to validation IoU (default=True).
    model_path : str, optional
        File path to save the best model (default="best_model.keras").

    Returns
    -------
    model : tensorflow.keras.Model
        The trained U-Net model.
    history : keras.callbacks.History
        Training history with loss and metric evolution.
    """
    
    print(f"📊 Shape of X_train: {X_train.shape}")
    print(f"📊 Shape of y_train: {y_train.shape}")
    print(f"🎯 Number of classes: {num_classes}")

    # -------- Preprocessing --------
    X = X_train.astype(np.float32, copy=True)
    y = y_train.astype(np.float32, copy=True)

    # 1) Replace NaN in X and y with specified null_value
    if fill_nulls:
        X[np.isnan(X)] = null_value
        y[np.isnan(y)] = null_value
        print("✅ Null values replaced with 0 in X and y")

    # 2) Normalize X only, band by band
    if normalize:
        patches, H, W, C = X.shape  # batch, height, width, channels
        for i in range(C):
            band = X[:, :, :, i]
            band_norm = StandardScaler().fit_transform(band.reshape(-1, 1)).reshape(patches, H, W)
            X[:, :, :, i] = band_norm
        print("✅ Normalization applied band by band with StandardScaler")

    # Quick check
    print('Values in labels, min: %d & max: %d' % (np.min(y), np.max(y)))

    # -------- Validation split --------
    X_val = y_val = None
    if validation_data is not None:
        X_val, y_val = validation_data
        X_val = X_val.astype(np.float32, copy=False)
        y_val = y_val.astype(np.float32, copy=False)
        print("✅ Using provided validation_data")
    elif validation_split and validation_split > 0.0:
        X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
        print(f"✅ Split: {len(X)} training, {len(X_val)} validation")

    # -------- Labels preparation --------
    def prepare_labels(y_arr, n_classes):
        if n_classes > 1:
            # Ensure integer labels in range 0..C-1 before one-hot encoding
            y_int = y_arr.astype("int32", copy=False)
            y_flat = y_int.reshape(-1)
            y_cat = to_categorical(y_flat, n_classes)  # (Npix, C)
            return y_cat.reshape(y_arr.shape[0], y_arr.shape[1], y_arr.shape[2], n_classes)
        else:
            # Binary: clip to [0,1] and keep single channel
            y_bin = np.clip(y_arr, 0.0, 1.0).astype(np.float32, copy=False)
            return y_bin

    y_cat = prepare_labels(y, num_classes)
    val_data = None
    if y_val is not None:
        y_val_cat = prepare_labels(y_val, num_classes)
        val_data = (X_val, y_val_cat)

    print(f"📊 Final X_train shape: {X.shape}")
    print(f"📊 Final y_train shape: {y_cat.shape}")
    if val_data is not None:
        print(f"📊 Final X_val shape: {X_val.shape}")
        print(f"📊 Final y_val shape: {y_val_cat.shape}")

    # -------- UNet model --------
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1); p1 = Dropout(dropout_rate)(p1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2); p2 = Dropout(dropout_rate)(p2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3); p3 = Dropout(dropout_rate)(p3)

    # Bottleneck
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)

    # Decoder
    u5 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5); c5 = Dropout(dropout_rate)(c5)

    u6 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6); c6 = Dropout(dropout_rate)(c6)

    u7 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7); c7 = Dropout(dropout_rate)(c7)

    # Output
    if num_classes == 1:
        out_channels = 1
        activation = 'sigmoid'
        loss = 'binary_crossentropy'
    else:
        out_channels = num_classes
        activation = 'softmax'
        loss = 'categorical_crossentropy'

    outputs = Conv2D(out_channels, 1, activation=activation)(c7)

    # -------- Compilation --------
    mean_iou_metric = make_mean_iou_metric(num_classes)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=loss,
                  metrics=['accuracy', mean_iou_metric]
    )
    print("✅ Model compiled successfully (monitor: val_mean_iou)")

    # -------- Callbacks (monitor IoU, mode 'max') --------
    callbacks = []
    if save_best_model and val_data is not None:
        checkpoint = ModelCheckpoint(
            model_path,
            monitor="val_mean_iou",
            mode="max",
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        print("💾 Saving best model based on val_mean_iou (max)")

    # -------- Training --------
    print("🚀 Starting training...")
    if data_augmentation:
        # Simple and safe augmentation for multiband images
        def gen(Xa, ya, bs):
            H, W = Xa.shape[1], Xa.shape[2]
            while True:
                idx = np.random.randint(0, len(Xa), size=bs)
                bx, by = Xa[idx].copy(), ya[idx].copy()
                # Flips
                if np.random.rand() < 0.5:
                    bx = np.flip(bx, axis=1); by = np.flip(by, axis=1)
                if np.random.rand() < 0.5:
                    bx = np.flip(bx, axis=2); by = np.flip(by, axis=2)
                # Rotations 0/90/180/270
                k = np.random.randint(0, 4)
                bx = np.rot90(bx, k=k, axes=(1, 2))
                by = np.rot90(by, k=k, axes=(1, 2))
                yield bx, by

        train_gen = gen(X, y_cat, batch_size)
        steps = max(1, len(X) // batch_size)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X, y_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

    return model, history

# ==========================================
# PREDICTION FUNCTIONS
# ==========================================
def predict_raster(model, raster_path, patch_size=256, num_classes=1,
                   output_path=None, fill_nulls=True, null_value=0, normalize=True,
                   overlap=0):
    """
    Predicts over a complete raster file and reconstructs the original image.

    Args:
        model: Trained model (U-Net)
        raster_path (str): Path to the .tif file for prediction
        patch_size (int): Patch size used during training
        num_classes (int): Number of classes of the model
        output_path (str): Path to save the predicted raster (optional)
        fill_nulls (bool): Whether to replace null values with null_value
        null_value: Value to replace nulls
        normalize (bool): Whether to apply normalization
        overlap (int): Number of overlapping pixels between patches

    Returns:
        np.array: Predicted reconstructed image
    """

    print(f"Processing file: {raster_path}")

    # Read raster metadata
    with rasterio.open(raster_path) as src:
        original_height, original_width = src.height, src.width
        transform = src.transform
        profile = src.profile
        nodata = src.nodata

    # Process raster into patches
    X_patches = processing(
        raster_path=raster_path,
        patch_size=patch_size,
        export_patches=False,
        overlap=overlap
    )

    # Handle null values in patches
    if fill_nulls:
        print("Handling null values...")
        X_patches[np.isnan(X_patches)] = null_value

        # Also handle infinities if present
        inf_mask = np.isinf(X_patches)
        X_patches[inf_mask] = null_value

    # Normalize if required
    if normalize:
        patches, H, W, C = X_patches.shape  # batch, height, width, channels
        for i in range(C):
            band = X_patches[:, :, :, i]  # all samples for band i
            band_norm = StandardScaler().fit_transform(band.reshape(-1, 1)).reshape(patches, H, W)
            X_patches[:, :, :, i] = band_norm
        print("✅ Normalization applied to X")

    # Perform prediction
    print("Running prediction...")
    y_pred = model.predict(X_patches, verbose=1)

    # Process predictions
    if num_classes > 1:
        # Multiclass: take the class with highest probability
        y_pred_classes = np.argmax(y_pred, axis=3)
        print(f"Predicted classes: {np.unique(y_pred_classes)}")
    else:
        # Binary: apply threshold
        y_pred_classes = (y_pred > 0.5).astype(np.uint8).squeeze()
        print(f"Predicted classes: {np.unique(y_pred_classes)}")

    # Reconstruct full image (with overlap if applicable)
    reconstructed = reconstruct_from_patches(
        y_pred_classes,
        original_height,
        original_width,
        patch_size,
        overlap=overlap
    )

    # Save result if specified
    if output_path is not None:
        save_prediction_raster(reconstructed, output_path, profile, transform)
        print(f"Prediction saved at: {output_path}")

    return reconstructed


def reconstruct_from_patches(patches, original_height, original_width, patch_size, overlap=0):
    """
    Reconstructs a full image from patches, averaging values in overlapped regions.
    """
    step = patch_size - overlap
    if step <= 0:
        raise ValueError("❌ Overlap must be smaller than patch_size.")

    n_patches_h = (original_height - overlap + step - 1) // step
    n_patches_w = (original_width - overlap + step - 1) // step

    full_height = (n_patches_h - 1) * step + patch_size
    full_width = (n_patches_w - 1) * step + patch_size
    sum_array = np.zeros((full_height, full_width), dtype=np.float32)
    count_array = np.zeros((full_height, full_width), dtype=np.float32)

    patch_idx = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start, x_start = i * step, j * step
            y_end, x_end = y_start + patch_size, x_start + patch_size

            if patch_idx >= len(patches):
                break

            patch_pred = patches[patch_idx].squeeze() if patches.ndim > 2 else patches[patch_idx]
            sum_array[y_start:y_end, x_start:x_end] += patch_pred
            count_array[y_start:y_end, x_start:x_end] += 1
            patch_idx += 1

    reconstructed = sum_array / np.maximum(count_array, 1)
    reconstructed = reconstructed[:original_height, :original_width]

    return reconstructed.astype(np.uint8)


def save_prediction_raster(prediction, output_path, original_profile, original_transform):
    """Saves prediction as a georeferenced raster."""
    prediction_profile = original_profile.copy()
    prediction_profile.update({
        'dtype': rasterio.uint8,
        'count': 1,
        'height': prediction.shape[0],
        'width': prediction.shape[1],
        'transform': original_transform,
        'nodata': 255
    })

    with rasterio.open(output_path, 'w', **prediction_profile) as dst:
        dst.write(prediction.astype(rasterio.uint8), 1)


def plot_prediction(prediction, title="Prediction", cmap="viridis"):
    """Plots prediction raster."""
    plt.figure(figsize=(10, 8))
    if prediction.ndim > 2:
        prediction = prediction.squeeze()
    im = plt.imshow(prediction, cmap=cmap)
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_comparison(prediction, reference, title="Prediction vs Reference"):
    """Plots side-by-side prediction vs reference raster."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if prediction.ndim > 2:
        prediction = prediction.squeeze()
    if reference.ndim > 2:
        reference = reference.squeeze()

    im1 = ax1.imshow(prediction, cmap="viridis")
    ax1.set_title("Prediction", fontsize=12)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(reference, cmap="viridis")
    ax2.set_title("Reference", fontsize=12)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# -


