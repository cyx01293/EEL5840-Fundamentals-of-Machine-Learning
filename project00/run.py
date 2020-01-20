# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 14:36:25 2018

@author: asus
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import neighbors
import train
import test
filename_train='data_train'
u,var=train.train(filename_train)
filename_test='test_sample'  #select 'data_test' or 'test_sample'
coordinate=test.test(filename_test,u,var)


#import imageio
#test_sample=imageio.imread('test_sample.png')