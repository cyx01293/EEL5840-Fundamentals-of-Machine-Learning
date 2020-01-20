Select 'test_sample' or 'data_test' in line 18 of run.py  to change to run sample image or complete image.

In train.py and test.py, select different gridnum from 10, 25, 50 and so on. The difference of them is illustrated in the project report.


### package needed:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors
import imageio

The problem statement can be seen [here](https://github.com/cyx01293/EEL5840-Fundamentals-of-Machine-Learning/blob/master/project00/project_00.pdf) and the report [here.](https://github.com/cyx01293/EEL5840-Fundamentals-of-Machine-Learning/blob/master/project00/project00_report.pdf)
