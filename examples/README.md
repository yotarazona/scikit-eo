<!-- #region -->
# Running notebooks
<!-- #region -->

> The image and signatures to be used can be downloaded [here](https://drive.google.com/drive/folders/193RhNpACu9THcOZu8OzMh-btnFCOgHrU?usp=sharing).

**There are three options to run the notebooks**:

- 1) *Locally*: You will need to download the data and save it in a folder (e.g., folder 'data'), then you can connect the folder with the following line as follows:
 
```python
# output folder ('data')
data_dir = ".../home/data"

# reading images and other files
path_raster = data_dir + "/" + "name_of_file.tif"
img = rasterio.open(path_raster))
```

- 2) *Connecting to Google Drive*: If your files are within Google Drive, you will only need to run the following line:

```python
from google.colab import drive
drive.mount('/content/drive')

# output folder ('data')
data_dir = "/content/drive/MyDrive/data/"

# reading images and other files
path_raster = data_dir + "/" + "name_of_file.tif"
img = rasterio.open(path_raster))
```