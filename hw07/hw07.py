# -*- coding: utf-8 -*-
"""
File:   hw07.py
Author: Alina Zare
Mod.by: Connor McCurley
Date:   2018-11-26
Desc:   Convert an image into a binary feature vector with the last element
        as the corresponding image label.
"""

"""
====================================================
================ Import Packages ===================
====================================================
"""
import sys

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt

"""
====================================================
================ Define Functions ==================
====================================================
"""

def process_image(in_fname,out_fname,debug=False):

    # load image
    x_in = np.array(Image.open(in_fname))

    # convert to grayscale
    x_gray = 1.0-rgb2gray(x_in)

    if debug:
        plt.figure(1)
        plt.imshow(x_gray)
        plt.title('original grayscale image')
        plt.show()

    # threshold to convert to binary
    thresh = filt.threshold_minimum(x_gray)
    fg = x_gray > thresh

    if debug:
        plt.figure(2)
        plt.imshow(fg)
        plt.title('binarized image')
        plt.show()

    # find bounds
    nz_r,nz_c = fg.nonzero()
    n_r,n_c = fg.shape
    l,r = max(0,min(nz_c)-1),min(n_c-1,max(nz_c)+1)+1
    t,b = max(0,min(nz_r)-1),min(n_r-1,max(nz_r)+1)+1

    # extract window
    win = fg[t:b,l:r]

    if debug:
        plt.figure(3)
        plt.imshow(win)
        plt.title('windowed image')
        plt.show()

    # resize so largest dim is 48 pixels 
    max_dim = max(win.shape)
    new_r = int(round(win.shape[0]/max_dim*48))
    new_c = int(round(win.shape[1]/max_dim*48))

    win_img = Image.fromarray(win.astype(np.uint8)*255)
    resize_img = win_img.resize((new_c,new_r))
    resize_win = np.array(resize_img).astype(bool)

    # embed into output array with 1 pixel border
    out_win = np.zeros((resize_win.shape[0]+2,resize_win.shape[1]+2),dtype=bool)
    out_win[1:-1,1:-1] = resize_win

    if debug:
        plt.figure(4)
        plt.imshow(out_win,cmap='Greys')
        plt.title('resized windowed image')
        plt.show()

    #save out result as numpy array
    np.save(out_fname,out_win)

"""
====================================================
========= Generate Features and Labels =============
====================================================
"""
#i = 1
#letter = ['a','b','c','d','h','i','j','k']
#num = [1,2,3,4,5,6,7,8]
#labels = []
#for l,k in zip(letter,num):
#    for i in range(1,11):
#        picture = l+str(i)+'.'+'png'
#        ndata=l+str(i)+'.'+'npy'
#        labels.append(k)
#        process_image(picture,ndata,debug=True)
#        
#data = []
#for l in letter:
#    for i in range(1,11):
#        name=l+str(i)+'.'+'npy'
#        tempdata=np.load(name)
#        plt.imshow(tempdata)
#        plt.show()
#        data.append(tempdata)
#
#        
#data=np.array(data)
#dataname="data.npy"
#labelsname="labels.npy"
#np.save(dataname,data)
#np.save(labelsname,labels)
#
#im01=Image.open("letter.jpg")
#im01.show()

i = 1
letter = ['a','b','c','d','h','i','j','k']
#letter = ['j']
num = [1, 2, 3, 4, 5, 6, 7, 8]
labels = []
for l,k in zip(letter, num):
    for i in range(1,11):
        picture = l + str(i) + '.'+ 'png'
        ndata = l + str(i) + '.'+ 'npy'
        labels.append(k)
        process_image(picture, ndata, debug=True)

data = []
for l in letter:
    for i in range(1,11):
        name = l + str(i) + '.'+ 'npy'        
        tempdata = np.load(name)
        plt.imshow(tempdata)
        plt.show()
        data.append(tempdata)
        
data = np.array(data)
dataname = "data.npy"
labelsname = "labels.npy"
np.save(dataname,data)
np.save(labelsname,labels)
#im01=Image.open("letter.jpg")
#im01.show()
#if __name__ == '__main__':
#
#    # To not call from command line, comment the following code block and use example below 
#    # to use command line, call: python hw07.py K.jpg output
#
#    if len(sys.argv) != 3 and len(sys.argv) != 4:
#        print('usage: {} <in_filename> <out_filename> (--debug)'.format(sys.argv[0]))
#        sys.exit(0)
#    
#    in_fname = sys.argv[1]
#    out_fname = sys.argv[2]
#
#    if len(sys.argv) == 4:
#        debug = sys.argv[3] == '--debug'
#    else:
#        debug = False


#    #e.g. use
#    process_image('C:/Desktop/K.jpg','C:/Desktop/output.npy',debug=True)
#process_image('j1.png','j1.npy')
