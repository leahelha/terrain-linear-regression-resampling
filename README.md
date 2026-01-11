# An Exploratory Research Into Linear Regression and Resampling Techniques for Studying Topographical Data:


## Project overview:
This is the code for reproducing our results in _Project 1_ of **FYS-STK4155** for the Autumn 2023 semester at UiO. It contains functions for making a prediction for the entire dataset after a model has been fitted with the training data, and functions for plotting and creating latex tables. The graphs from the plotting are stored in the _plots_ folder while the Latex tables are stored in the _tables_ folder.

Also for our Lasso regression we used the defualt iterations of $1000$ iterations.


## Installation instructions:
To install all the necessary packages, run this code:

```Python
pip install -r requirements.txt
```

where **requirements.txt** contains all the required packages to run the code for this repository.


## Datasets:
There are two dataset used, one from real data and one from syntizhed data.

### Syntizhed data:
We use a syntizhed dataset that is made using the _Franke function_. The function `def FrankeFunction` is defined in **reg_class.py**.

### Real data:
The dataset is digital terrain data of Norway. It is the file **SRTM_data_Norway_1.tif** and can be found at:

[https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project1/DataFiles/SRTM_data_Norway_1.tif
](https://github.com/CompPhysics/MachineLearning/blob/master/doc/Projects/2023/Project1/DataFiles/SRTM_data_Norway_1.tif
)



## Usage guide:
If you run the file **topological.py**, you run the analysis and plotting on topograpical data:

```Python
python3 topological.py
```

while if you run the file **reg_class.py** as a script:

```Python
python3 reg_class.py
```

the analysis and plotting will be on the synthized dataset (made from the _Franke function_).

The files **cross_val.py** and **cross_val_debug.py** were created for developing and debugging the cross validation code in **reg_class.py** and are irrelevant for the analysis. However to run the other resampling method, _bootstrap_, you need to run:

```Python
python3 ec3.py
```

Lastly **test_reg_class.py** is where we have tested that our code runs correctly. To do these tests, run the following:

```Python
python3 test_reg_class.py
```
