<!-- #region -->
# **Installation**

To use **scikit-eo** it is necessary to install it in your terminal. There are two options to use its functions/classes:

### 1. From PyPI

**scikit-eo** is available on [PyPI](https://pypi.org/project/scikeo/), so to install it, run this command in your terminal:

```python
pip install scikeo
```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.


## 2. Installing from source

It is also possible to install the latest development version directly from the GitHub repository with:

```python
pip install git+https://github.com/yotarazona/scikit-eo
```
This is the preferred method to install scikit-eo, as it will always install the most recent stable release.

## containerizing ```scikit-eo```

**Note**: It is a recommended practice to provide some instructions for isolating/containerizing ```scikit-eo```. It would benefit their use and thus avoid that some dependencies are not compatible with others. For example, conda provides an easy solution.

```python
conda create -n scikiteo python = 3.8
```
Then, activate the environment created

```python
conda activate scikiteo
```
Then finally, ```scikit-eo``` can be install within this new environment using via PyPI or from the GitHub repository.

<!-- #endregion -->
