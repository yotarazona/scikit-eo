#!/usr/bin/env python
# -*- coding: utf-8 -*-
# +
""" Tests for the scikit-eo package. """

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
import pandas as pd
from scikeo.process import confintervalML
from scikeo.sma import sma
from scikeo.pca import PCA
from scikeo.tassCap import tassCap
from scikeo.fusionrs import fusionrs
from scikeo.process import crop
from scikeo.rkmeans import rkmeans
from scikeo.mla import MLA
from scikeo.process import extract
from dbfread import DBF
from scikeo.calmla import calmla
from scikeo.deeplearning import DL
from scikeo.calkmeans import calkmeans


def test_confintervalML():
    
    """Confusion Matrix by Estimated Proportions of area an uncertainty is tested. 
    
    To carry out it, this function was tested using ground-truth values obtained by 
    Olofsson et al. (2014).
    """
    conf_error = pd.read_csv("tests/data/confusion_matrix.csv", index_col= 0, sep = ';')
    # only confusion matrix values
    values = conf_error.iloc[0:4,0:4].to_numpy()
    # number of pixels for each class
    img = np.array([200000, 150000, 3200000, 6450000])
    res = confintervalML(matrix = values, image_pred = img, pixel_size = 30, nodata = -9999)
    assert round(res.get('Overall_accuracy'),3) == 0.947
    
def test_sma():
    """Spectral mixture analysis is tested."""
    # image to be processed
    img = rasterio.open('tests/data/LC08_232066_20190727.tif')
    # endmembers
    endm =[[8980,8508,8704,13636,16579,11420], # soil
            [8207,7545,6548,16463,9725,6673], # forest
            [9276,9570,10089,6743,5220,5143], # water
           ]
    endm = np.array(endm)
    # applying the sma function
    frac = sma(image = img, endmembers = endm)
    assert frac.shape[2] == 3
    
def test_pca():
    """Principal Component Analysis is tested."""
    # image to be processed
    img = rasterio.open('tests/data/LC08_232066_20190727.tif')
    # Applying the PCA function:
    arr_pca = PCA(image = img, stand_varb = True)
    img_pca = arr_pca.get('PCA_image')
    assert img_pca.shape[2] == 6

def test_tassCap():
    """Tasseled-Cap is tested."""
    # image to be processed
    img = rasterio.open('tests/data/LC08_232066_20190727.tif')
    # Applying the tassCap function:
    arr_tct = tassCap(image = img, sat = 'Landsat8OLI')
    assert arr_tct.shape[2] == 3

def test_fusionrs():
    """Fusion optical and radar satelite images is tested."""
    # image to be processed
    # optical
    optical = rasterio.open('tests/data/LC08_003069_20180906_clip.tif')
    # radar
    radar = rasterio.open('tests/data/S1_2018_VV_VH_clip.tif')
    # Applying the fusionrs function:
    fusion = fusionrs(optical = optical, radar = radar)
    # Cumulative variance (%)
    cum_var = fusion.get('Cumulative_variance')*100
    assert round(cum_var[8],0) == 100

def test_crop():
    """cliping a satellite images is tested."""
    # raster to be clipped
    path_raster = "tests/data/LC08_232066_20190727.tif"
    # area of Interes -> shapefile
    path_shp = "tests/data/clip.shp"
    # Path where the image will be saved
    output_path_raster = "tests/data"
    # The raster name
    output_name = 'LC08_232066_20190727_clip'
    # Applying the crop() function:
    crop(image = path_raster, vector = path_shp, 
         filename = output_name, 
         filepath = output_path_raster)
    clip_image = rasterio.open(output_path_raster + '/' + output_name+ '.tif')
    assert type(clip_image) == rasterio.io.DatasetReader

def test_kmeans():
    """k-means classification is tested."""
    # image to be processed
    img = rasterio.open('tests/data/LC08_232066_20190727.tif')
    # Applying rkmeans() algorithm with four classes:
    arr_rkmeans = rkmeans(image = img, k = 4, max_iter = 300)
    assert type(arr_rkmeans) == np.ndarray

def test_machineLearning():
    """Machine learning algorithms are tested."""
    # image to be processed
    path_raster = "tests/data/LC08_232066_20190727.tif"
    img = rasterio.open(path_raster)
    # endmembers
    path_endm = "tests/data/endmembers.dbf"
    endm = DBF(path_endm)
    endm = pd.DataFrame(endm)
    # Instance of mla():
    inst = MLA(image = img, endmembers = endm)
    # Applying Random Forest with 60% of data to train
    rf_class = inst.RF(training_split = 0.6)
    assert rf_class.get('Overall_Accuracy') >= 0.7
    
def test_CalibratingMachineLearning():
    """Calibrating machine learning algorithms are tested."""
    # endmembers
    path_endm = "tests/data/endmembers.dbf"
    endm = DBF(path_endm)
    endm = pd.DataFrame(endm)
    # Instance of Calmla():
    inst = calmla(endmembers = endm)
    # Instance of splitData():
    data = inst.splitData()
    # Calibrating with Monte Carlo Cross-Validation Calibration (MCCV)
    error_mccv = inst.MCCV(split_data = data, models = ('svm', 'dt', 'rf', 'nb'), n_iter = 10)
    # error of Random Forest
    error_rf = np.mean(np.array(error_mccv.get('rf')))
    # error of Decision Tree
    error_dt = np.mean(np.array(error_mccv.get('dt')))
    assert error_dt > error_rf
    
def test_DeepLearning():
    """Deep Learning is tested."""
    # image to be processed
    path_raster = "tests/data/LC08_232066_20190727.tif"
    img = rasterio.open(path_raster)
    # endmembers
    path_endm = "tests/data/endmembers.dbf"
    endm = DBF(path_endm)
    endm = pd.DataFrame(endm)
    # Instance of DL():
    inst = DL(image = img, endmembers = endm)
    # Applying the FullyConnected() function of Deep Learning:
    fc = inst.FullyConnected(hidden_layers = 4, 
                         hidden_units = [64,16,8,8], 
                         output_units = 4,
                         input_shape = (6,), 
                         epochs = 100, 
                         batch_size = 32, 
                         training_split = 0.8)
    
    assert fc.get('Overall_Accuracy') >= 0.6

def test_calkmeans():
    # Create synthetic image data
    width, height, bands = 100, 100, 3  # Image dimensions and number of bands
    np.random.seed(42)  # For reproducibility

    # Generate synthetic image with three clusters
    cluster_centers = np.array([
        [50, 50, 50],
        [150, 150, 150],
        [250, 250, 250]
    ])

    total_pixels = height * width  # 10000 pixels
    num_clusters = len(cluster_centers)  # 3 clusters

    # Calculate base size and remainder
    base_size = total_pixels // num_clusters  # 3333 pixels per cluster
    remainder = total_pixels % num_clusters   # 1 pixel remaining

    # Initialize cluster sizes
    cluster_sizes = [base_size] * num_clusters  # [3333, 3333, 3333]

    # Distribute the remainder
    for i in range(remainder):
        cluster_sizes[i] += 1  # Now cluster_sizes = [3334, 3333, 3333]

    cluster_data = []
    for center, size in zip(cluster_centers, cluster_sizes):
        data = np.random.normal(loc=center, scale=10, size=(size, bands))
        cluster_data.append(data)

    # Combine and shuffle the data
    image_data = np.vstack(cluster_data)
    np.random.shuffle(image_data)
    image_data = image_data.reshape((height, width, bands)).astype(np.float32)

    # Introduce nodata values
    nodata_value = -99999.0
    image_data[0, 0, :] = nodata_value  # Set the first pixel to nodata

    # Create an in-memory raster dataset
    transform = rasterio.transform.from_origin(0, 0, 1, 1)  # Arbitrary transform
    with MemoryFile() as memfile:
        # Open the in-memory file in write mode
        with memfile.open(
            driver='GTiff',
            height=height,
            width=width,
            count=bands,
            dtype='float32',
            transform=transform,
            nodata=nodata_value
        ) as dataset_writer:
            # Write data to the dataset
            for i in range(bands):
                dataset_writer.write(image_data[:, :, i], i + 1)
            # Note: No need to call dataset_writer.close() explicitly within 'with' block

        # Now reopen the dataset in read mode
        with memfile.open() as dataset:
            # dataset_reader is a DatasetReader

            # Run the calkmeans function
            # Case 1: Find the optimal number of clusters
            results_optimal_k = calkmeans(
                image=dataset,
                k=None,
                algo=('lloyd', 'elkan'),
                n_iter=5,
                nodata=nodata_value
            )

            # Case 2: Evaluate algorithms for a given k
            k_value = 3
            results_given_k = calkmeans(
                image=dataset,
                k=k_value,
                algo=('lloyd', 'elkan'),
                n_iter=5,
                nodata=nodata_value
            )

            # Case 2: Evaluate algorithms for a given k
            k_value = 3
            results_given_k = calkmeans(
                image=dataset,
                k=k_value,
                algo=('lloyd', 'elkan'),
                n_iter=5,
                nodata=nodata_value
            )

    # Assertions to check the outputs
    # Case 1: Check if results contain expected keys and correct number of inertia values
    assert isinstance(results_optimal_k, dict), "Results should be a dictionary."
    for algorithm in ('lloyd', 'elkan'):
        assert algorithm in results_optimal_k, f"Algorithm '{algorithm}' should be in results."
        inertias = results_optimal_k[algorithm]
        assert len(inertias) == 5, "Inertia list should have length equal to n_iter."
        assert all(isinstance(val, float) for val in inertias), "Inertias should be floats."

    # Case 2: Check if results contain expected keys and correct number of inertia values
    assert isinstance(results_given_k, dict), "Results should be a dictionary."
    for algorithm in ('lloyd', 'elkan'):
        assert algorithm in results_given_k, f"Algorithm '{algorithm}' should be in results."
        inertias = results_given_k[algorithm]
        assert len(inertias) == 5, "Inertia list should have length equal to n_iter."
        assert all(isinstance(val, float) for val in inertias), "Inertias should be floats."

    print("All tests passed.")
