
import os
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
import math as mt
import cv2
import tensorflow as tf
from operator import itemgetter

IMAGE_FILE = 'artificialset_128.npy'
LABELS_FILE = 'artificial_labels_128.npy'
RAYS_FILE = 'artificial_rays128.npy'

TEST_IMAGE_FILE = 'testset_128.npy'
TEST_LABELS_FILE = 'test_labels_128.npy'
TEST_RAYS_FILE = 'test_rays128.npy'

def generateArtificialSet(width, height, psize_min, psize_max, num):

    #Coordinate X,Y di un'immagine width x height
    X, Y = np.meshgrid(np.arange(width), np.arange(height))   
    #Inizializzo i pixels a bianco (255)
    I = 255 * np.ones((num, width, height), dtype=np.uint8) 
    #Inizializzo i raggi del disco (pupilla) al quadrato e le coordinate random per il centro della pupilla
    R2 = rd.randint(psize_min, psize_max, (num))**2 
    xy = rd.randint(-12, 13, (num, 2))   
    #Setto il valore dei pixels all'interno del disco a 0 (nero)
    for i in range(len(I)):
        I[i][((X - 64 - xy[i][0])**2 + (Y - 64 - xy[i][1])**2)< R2[i]] = 0
    #Restituisco l'immagine, le coordinate del centro della pupilla, e il quadrato del raggio della pupila
    return I, xy, R2

        
        
def createModel(image_shape = (128, 128, 1), kernel_size = (5, 5)):
    """
    Create a model of Convolutional Neural Network, with 5 layers of convolution.

    image_shape (tuple) --> tuple containing the dimensions of the images. Tensorflow expect RGB images, 
                            so the tuple must be (n,m,3). ---Default value is (128, 128, 1)---
    kernel_size (tuple) --> kernel size for convolution.  ---Default value is (5,5)---
    """
    
    model = models.Sequential()
    model.add(layers.Conv2D(12, kernel_size, activation = 'relu', padding='same', input_shape= image_shape))
    model.add(layers.Conv2D(12, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.Conv2D(12, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.Conv2D(12, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.Conv2D(12, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(2))

    model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

    print(model.summary())
    return model


def trainModel(model, train_images, train_labels, test_images, val_labels,  n_epoch= 10, filename='void'):
    """
    Train the model by associating the label to the respective image.

    model (tensorflow.object) --> an architecture of an DNN
    train_images (3D-array)   --> set of images
    train_labels (1D-array)   --> set of labels
    n_epoch (int)             --> Is an iper-parameter. An epoch is one complete presentation of the data set 
                                  to be learned from the DNN. Default value is 10.
    filename (string)         --> Name of the .h5 file wich contain the trained model. Default value = 'void'
                                  If is necessary to save the model, just give a filename as argument (!= void)
    """
    history = model.fit(train_images, train_labels, epochs = n_epoch, validation_data=(test_images, val_labels))
    #history = model.fit(train_images, train_labels, epochs = n_epoch)

    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

    if filename != 'void':
        model.save(filename)

def predictImages(model, test_images):
    """
    Predict the associate label of an image by using a DNN model.
    Return an array of predictions.
    """
    return model.predict(test_images)

def histogram(pred, real):
    """
    Return the histogram of the distance between prevision and real value.
    predictions (2D array) --> results of the previsions of a DNN.
    read_out (string)      --> define the read out method (default value 'classification'). 
                               'classification' , take the maximum value of a prediction. (used for classifications)
                               'weighted', compute the weighted average of a prediction. A prediction is an array
                                containing the score of each label. The top score label predict will have the highest 
                                weight, the second scor the second highest weight, and so on. (used for regression)
    """
    err = abs(pred - real)
    dist = np.sqrt(abs(pred[:, 0] - real[:, 0])**2 + abs(pred[:, 1] - real[:, 1])**2)
    plt.subplot(1, 3, 1)
    plt.hist(err[:,0])
    plt.xlabel('Distance (abs) between x_prev and x_real')
    plt.ylabel('Number of point')
    plt.title('Error PREV-REAL X-COORD')
    plt.subplot(1, 3, 2)
    plt.hist(err[:,1])
    plt.xlabel('Distance (abs) between y_prev and y_real')
    plt.ylabel('Number of point')
    plt.title('Error PREV-REAL Y-COORD')
    plt.subplot(1, 3, 3)
    plt.hist(dist)
    plt.xlabel('Distance (abs) between prev and real')
    plt.ylabel('Number of point')
    plt.title('Error PREV-REAL')
    plt.show()



if __name__ == '__main__':
    #CREATE A NEW MODEL
    model = createModel()
    #GENERATE TRAIN SET, TEST SET, VALIDATION SET. SET --> (IMAGES, LABEL [X,Y], RAY)
    train_images, train_labels, train_rays = generateArtificialSet(128, 128, 2, 21, 10000)
    test_images, test_labels, test_rays = generateArtificialSet(128, 128, 2, 21, 3000)
    val_images, val_labels, val_rays = generateArtificialSet(128, 128, 2, 21, 2000)
    #CHANGE RANGE FROM [0-255] TO [0-1]
    train_images, test_images, val_images = train_images / 255.0, test_images / 255.0, val_images / 255.0
    #RESHAPE
    train_images = np.reshape(train_images, (train_images.shape[0], 128, 128, 1))
    test_images = np.reshape(test_images, (test_images.shape[0], 128, 128, 1))
    val_images = np.reshape(val_images, (val_images.shape[0], 128, 128, 1))
    #TRAIN MODEL
    trainModel(model, train_images, train_labels, test_images, test_labels, n_epoch = 2, filename = 'prova_1.h5')
    #-----IF THE MODEL IS ALREADY TRAINED, UNCOMMENT NEXT LINE AND COMMENT NEXT ONE------#
    #model = tf.keras.models.load_model('esperimento_convnet_1.h5')
    prediction = predictImages(model, test_images)
    histogram(prediction, test_labels)
