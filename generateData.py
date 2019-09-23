# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:21:33 2018

@author: SmartDATA
"""
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import skimage.filters as filt



'''============================================================================'''
'''============================================================================'''


image_new = np.load('img/original image/ClassData.npy')
label_new = np.load('img/original image/ClassLabels.npy') * 1.0
image_old = np.load('img/original image/TrainImages.npy') 
label_old = np.load('img/original image/TrainY.npy')
label_old = np.reshape(label_old,(1655)) * 1.0
image = np.concatenate((image_new, image_old),axis=0) 
label = np.concatenate((label_new, label_old))
'''============================================================================'''
'''============================================================================'''
imageSize = [32,32]

reshaped_image = []
for i in range(len(image)):
    im = np.array(image[i])
    im = Image.fromarray(im.astype(np.uint8))
    img = im.resize((imageSize[0],imageSize[1]))
    img = np.array(img)
    reshaped_image.append(img)
    if i%10 == 0:        
        plt.imshow(img)
        plt.show()

reshaped_image = np.array(reshaped_image)
print(reshaped_image.shape)
print(label.shape)

'''============================================================================'''
'''============================================================================'''
imageName = 'img/reshaped image/image' + str(imageSize[0]) +'.npy'
labelName = 'img/reshaped image/label.npy'
#np.save('img/reshaped image/reshaped_image_new', reshaped_image)
np.save(imageName, reshaped_image)
np.save(labelName, label)


'''============================================================================'''
'''============================================================================'''
imageName_ab = 'img/reshaped image/image_ab' + str(imageSize[0]) +'.npy'
labelName_ab = 'img/reshaped image/label_ab.npy'

label_a = label[np.where(label == 1)]
label_b = label[np.where(label == 2)]
label_ab = np.concatenate((label_a, label_b)) 


image_a = reshaped_image[np.where(label == 1)]
image_b = reshaped_image[np.where(label == 2)]
image_ab = np.concatenate((image_a, image_b),axis=0)
np.save(imageName_ab, image_ab)
np.save(labelName_ab, label_ab)

'''============================================================================'''
'''============================================================================'''
fileNumber = 62
imageNumber = 55
label_allChar = []
for i in range(1, fileNumber+1):
    for j in range(1, imageNumber+1):
        in_fname = 'img/Sample'+ format(i, '03d') + '/' + format(i, '03d') + '-' + format(j, '03d') + '.png'
        out_fname = 'img_new/'+ format(i, '03d') + '-' + format(j, '03d') + '.npy'
        print(in_fname)
        process_image(in_fname,out_fname,debug=True)
        label_allChar.append(i)
        

label_allChar = np.array(label_allChar)
np.save('label_allChar.npy',label_allChar)


image_allChar = []

for i in range(1, fileNumber+1):
    for j in range(1, imageNumber+1):
        filePath = 'img_new/'+ format(i, '03d') + '-' + format(j, '03d') + '.npy'
        tempImage = np.load(filePath) * 1
        image_allChar.append(tempImage)

image_allChar = np.array(image_allChar)
np.save('image_allChar.npy',image_allChar)










