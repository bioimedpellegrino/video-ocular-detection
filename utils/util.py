'''
Created on 07/11/2019

@author: ap
'''

from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
import os.path
import sys
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




#DEFINE NAME OF .CSV FILE
FILE_NAME  = 'Training_512.dat'


def readFileCSV(filename):
    """
    Scan a csv file and return it as a numpy array
    """
    return np.asarray(pd.read_csv(filename, sep="\t"))


def getImage(path, width, height, resize = False, resize_w = 256, resize_h = 256):
    """
    Gets an image from file path, crops it according to his viewport
    and return it as a numpy array.
    If images resolution is too high, is possibile to downsize by using arguments "resize, resize_w, resize_h".

    path (string);
    width (int), height(int): viewport of the image;
    resize (bolean): if true, resize the image according to resize_w, resize_h
    resize_w, resize_h (int): new dimensions of image
    """
    #CREATE AN EMPTY IMAGE Width x Height
    output = np.zeros((height, width))
    #GET THE ORIGINAL IMAGE 
    image = cv2.imread(path)
    #COMPUTE THE DIFFERENCE BETWEEN THE ORIGINAL SIZE AND THE VIEWPORT COMPUTE IN THE DATASET
    w = image.shape[1] - width
    h = image.shape[0] - height
    #CROP ORIGINAL IMAGE ACCORDING TO VIEWPORT
    output = image[h:image.shape[0], w:image.shape[1]]
    #DOWNSIZE IMAGE IF IS NECESSARY
    if resize:
        output = cv2.resize(output, (resize_w,resize_h))
    #AND RETURN IT
    return output 


def createFileTrainingSet(filename_images, filename_label):
    """
    Create a .npy file containing all the images (as matrix) ready to be loaded for training

    filename_images(string): name of the .npy images file
    filename_label(string): name of the .npy labels file
    """
    #LOADING THE CSV FILE
    dataset = readFileCSV(FILE_NAME)
    #GET THE TRAIN IMAGES (normalized to range 0-1) AND LABELS AS TUPLE
    #train_images = np.asarray(tuple(getImage(i[0], i[-2], i[-1], True, 128, 128) for i in dataset))
    #train_images = np.asarray(tuple(getROI(i[0], i[-2], i[-1]) for i in dataset))
    train_label_x = np.asarray(tuple(i[2] for i in dataset))
    train_label_y = np.asarray(tuple(i[3] for i in dataset))
    #SAVE IN A .npy ARCHIVE
    #filename_images = filename_images + '.npy'
    filename_label_x = filename_label + '_x' + '.npy'
    filename_label_y = filename_label + '_y' + '.npy'
    #np.save(filename_images, train_images)
    np.save(filename_label_x, train_label_x)
    np.save(filename_label_y, train_label_y)



def loadTrainingSet(filename_images, frac=0.8):
    """
    Return the images set ready for training the DNN.

    It return:
    train_images --> as numpy arrays
    test_images  --> as  numpy arrays

    filename_images (string): Name of .npy file containing the images (as 2D-array)
    
    frac: Is the percentage of how split the dataset in train_images & test_images.
          By default is 0.8, it means that 80% of images are for training
          and the remaining 20% is for testing.
    """
    if frac < 0 or frac > 1:
        raise ValueError("Percentage (frac) must be a number between 0 and 1")
    
    #LOAD LABELS
    images = np.load(filename_images)
    #GET LENGHT OF TRAINING SET
    lenght_train = int(len(images) * frac)
    #SPLIT THE DATASET IN TWO: TRAIN_SET AND TEST_SET ACCORDIND TO THE PERCENTAGE  
    train_images = images[0:lenght_train]
    test_images = images[lenght_train:]
    #AND RETURN THEM
    return (train_images), (test_images)

def loadLabels(filename_labels, frac=0.8):
    """
    Return labels of the image set.

    It return:
    train_labels --> as numpy arrays
    test_labels  --> as  numpy arrays

    filename_labels (string): Name of .npy file containing the labels (array)
    
    frac: Is the percentage of how split the dataset in train_labels & test_labels.
          By default is 0.8, it means that 80% of labels are for training
          and the remaining 20% is for testing.
    """

    #LOAD LABELS
    labels = np.load(filename_labels)
    #GET LENGHT OF TRAINING SET
    lenght_train = int(len(labels) * frac)
    #SPLIT THE LABLES IN TWO: TRAIN_SET AND TEST_SET ACCORDIND TO THE PERCENTAGE  
    train_labels = labels[0:lenght_train]
    test_labels = labels[lenght_train:]
    #AND RETURN THEM
    return (train_labels), (test_labels)

def rescaleLabels(labels, old_min, old_max, new_min, new_max):
    """
    Convert one range of numbers to another, maintaining ratio. 
    labels (1D-array)       --> Set of number to rescale
    new_min (int)           --> lower boundary of new range 
    new_max (int)           --> upper boundary of new range
    """

    old_range = old_max - old_min
    new_range = new_max - new_min
    output = np.zeros_like(labels)
    for i in range(len(labels)):
        output[i] = int((((labels[i] - old_min) * new_range) / old_range)) + new_min
    return output


def getROI(path, width, height, roi_w = 128, roi_h = 128):
    """
    Gets an image from file path, crops it according to his viewport,
    and return the region of the pupil.

    path (string);
    width (int), height(int): viewport of the image;
    roi_w: width of the ROI to select
    roi_h: height of the ROI to select
    """
    #CREATE AN EMPTY IMAGE Width x Height
    im = np.zeros((height, width))
    #CREATE AN EMPTY IMAGE roi_w x roi_h
    output = np.zeros((height, width))
    #GET THE ORIGINAL IMAGE 
    image = cv2.imread(path)
    #COMPUTE THE DIFFERENCE BETWEEN THE ORIGINAL SIZE AND THE VIEWPORT COMPUTE IN THE DATASET
    w = image.shape[1] - width
    h = image.shape[0] - height
    #CROP ORIGINAL IMAGE ACCORDING TO VIEWPORT
    im = image[h:image.shape[0], w:image.shape[1]]
    #SELECT ONLY THE PUPIL REGION
    w_start = int(width/2 - roi_w/2)
    w_end = int(width/2 + roi_w/2)
    h_start = int(height/2 - roi_h/2)
    h_end = int(height/2 + roi_h/2)

    output = im[w_start:w_end, h_start:h_end]
    #AND RETURN IT
    return output


def combineLabel(label_x, label_y):
    labels = []

    for i in range(len(label_x)):
        labels.append([label_x[i], label_y[i]])

    return labels

def bland_altman_plot(data1, data2):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    print(md, sd)
    plt.scatter(mean, diff, s = 20, alpha= 0.1)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.show()

def regression(prediction, real):
    x1 = prediction[0] - prediction[-1]
    x2 = real[0] - real[-1]
    pend = x2/x1
    b = real[-1]/(prediction[-1]*pend)
    fig, ax = plt.subplots()
    ax.scatter(prediction, real, s=10, alpha=0.3)
    ax.plot([0, 1000],[b, 1000], color='red')
    plt.xlim(600, 900)
    plt.ylim(600, 900)
    plt.show()

def cosineSimilarity(train_labels, n_neurons, size, sigma):
    X = np.linspace(-size/10, size/10, n_neurons)
    labels = []

    for j in range(len(train_labels)):
        L = np.zeros(n_neurons)
        for i in range(n_neurons):
            L[i] = np.exp(-(X[i] - train_labels[j])**2/(2*sigma**2))
        labels.append(L)    
    
    return np.asarray(labels)
    

def gaussWeights(labels, N, sz, sigma):
   X = np.linspace(-sz/10, sz/10, N).reshape((1, N))
   Y = labels.reshape((np.size(labels), 1))
   return np.exp(-(X-Y)**2/(2*sigma**2))


'''
if __name__ == '__main__':
    createFileTrainingSet('Train_Set/train_images', 'Train_Set/train_label')
'''

