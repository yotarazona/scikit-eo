<!-- #region -->
# Running notebooks tutorials

Before running the notebooks tutorials, datasets to be used must be downloaded. Basically, there are three options to run notebooks tutorials. 
<!-- #region -->

> The image and signatures to be used can be downloaded [here](https://drive.google.com/drive/folders/193RhNpACu9THcOZu8OzMh-btnFCOgHrU?usp=sharing).

**Options tu run notebook tutorials**:

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

- 3) *Read directly*: Reading dataset directly from the external server (GitHub in this case). To do this, ```requests```, ```zipfile``` and ```io``` will be installed before. Then you will only need to run the following line:h_raster)).

```python
import requests, zipfile
from io import BytesIO

# Defining the zip file URL
url = 'https://github.com/yotarazona/data/raw/main/data/01_machine_learning.zip'

# Split URL to get the file name
filename = url.split('/')[-1]

# Downloading the file by sending the request to the URL
req = requests.get(url)

# extracting the zip file contents
file = zipfile.ZipFile(BytesIO(req.content))
file.extractall()
```

**!The third option is part of each notebook tutorial, so that reading and executing the tutorials are automated as much as possible**¡.

Each Notebook has a dataset to be used and it is available in this link:
- [https://github.com/yotarazona/data](https://github.com/yotarazona/data)