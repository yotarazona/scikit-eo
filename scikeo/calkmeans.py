# -*- coding: utf-8 -*-
# +
from typing import Optional, Tuple, Dict, List
from sklearn.cluster import KMeans
import rasterio
import numpy as np

def calkmeans(
    image: rasterio.io.DatasetReader,
    k: Optional[int] = None,
    algo: Tuple[str, ...] = ("lloyd", "elkan"),
    max_iter: int = 300,
    n_iter: int = 10,
    nodata: float = -99999.0,
    **kwargs
) -> Dict[str, List[float]]:
    '''
    Calibrating KMeans Clustering Algorithm

    This function calibrates the KMeans algorithm for satellite image classification.
    It can either find the optimal number of clusters (k) by evaluating the inertia
    over a range of cluster numbers or determine the best algorithm ('lloyd' or 'elkan')
    for a given k.

    Parameters:
        image (rasterio.io.DatasetReader): Optical image with 3D data.
        k (Optional[int]): The number of clusters. If None, the function finds the optimal k.
        algo (Tuple[str, ...]): Algorithms to evaluate ('lloyd', 'elkan').
        max_iter (int): Maximum iterations for KMeans (default 300).
        n_iter (int): Number of iterations or clusters to evaluate (default 10).
        nodata (float): The NoData value to identify and handle in the data.
        **kwargs: Additional arguments passed to sklearn.cluster.KMeans.

    Returns:
        Dict[str, List[float]]: A dictionary with algorithm names as keys and lists of
                                inertia values as values.

    Notes:
        - If k is None, the function evaluates inertia for cluster numbers from 1 to n_iter.
        - If k is provided, the function runs KMeans n_iter times for each algorithm to
          evaluate their performance.
        - The function handles NoData values using fuzzy matching to account for floating-point precision.
    '''
    if not isinstance(image, rasterio.io.DatasetReader):
        raise TypeError('"image" must be raster read by rasterio.open().')

    bands: int = image.count
    rows: int = image.height
    cols: int = image.width

    # Read and reshape the image data
    st: np.ndarray = image.read()  # Shape: [bands, rows, cols]
    st_reorder: np.ndarray = np.moveaxis(st, 0, -1)  # Shape: [rows, cols, bands]
    arr: np.ndarray = st_reorder.reshape((rows * cols, bands))  # Shape: [rows*cols, bands]

    # Handle nodata values with fuzzy matching
    if np.isnan(nodata):
        nodata_mask = np.isnan(arr).any(axis=1)
    else:
        nodata_mask = np.isclose(arr, nodata).any(axis=1)

    # Extract valid data (rows without nodata)
    valid_data: np.ndarray = arr[~nodata_mask]

    # Check if there is any valid data
    if valid_data.size == 0:
        raise ValueError("No valid data found to perform clustering.")

    # Validate algorithms
    valid_algorithms = ("lloyd", "elkan")
    for algorithm in algo:
        if algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm '{algorithm}'. Must be one of {valid_algorithms}.")

    results: Dict[str, List[float]] = {}

    if k is None:
        # Finding the optimal number of clusters
        for algorithm in algo:
            inertias: List[float] = []
            for n_clusters in range(1, n_iter + 1):
                kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, algorithm=algorithm, **kwargs)
                kmeans.fit(valid_data)
                inertias.append(kmeans.inertia_)
            results[algorithm] = inertias
        return results

    elif isinstance(k, int) and k > 0:
        # Evaluating algorithms for a given k
        for algorithm in algo:
            inertias: List[float] = []
            for _ in range(n_iter):
                kmeans = KMeans(n_clusters=k, max_iter=max_iter, algorithm=algorithm, **kwargs)
                kmeans.fit(valid_data)
                inertias.append(kmeans.inertia_)
            results[algorithm] = inertias
        return results

    else:
        raise TypeError(f"'k' must be None or a positive integer. Got {k} of type {type(k)}.")
