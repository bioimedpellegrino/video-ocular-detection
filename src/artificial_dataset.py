
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
import util
from operator import itemgetter
IMAGE_FILE = 'artificialset_128.npy'
LABELS_FILE = 'artificial_labels_128.npy'
RAYS_FILE = 'artificial_rays128.npy'

TEST_IMAGE_FILE = 'testset_128.npy'
TEST_LABELS_FILE = 'test_labels_128.npy'
TEST_RAYS_FILE = 'test_rays128.npy'

size, N = 128, 20

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
    Create a model of Convolutional Neural Network, with 4 layers of convolution.

    image_shape (tuple) --> tuple containing the dimensions of the images. Tensorflow expect RGB images, 
                            so the tuple must be (n,m,3). ---Default value is (64, 64, 3)---
    kernel_size (tuple) --> kernel size for convolution.  ---Default value is (3,3)---
    """
    
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size, activation = 'relu', padding='same', input_shape= image_shape))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(16, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Conv2D(32, kernel_size, activation = 'relu', padding='same'))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(N, activation='softmax'))

    model.compile(optimizer='adam',
              loss='cosine_similarity',
              metrics=['cosine_similarity'])

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

def histogram2(predictions_x, predictions_y, real_x, real_y, read_out = 'top'):
    """
    Return the histogram of the distance between prevision and real value.
    predictions (2D array) --> results of the previsions of a DNN.
    read_out (string)      --> define the read out method (default value 'classification'). 
                               'classification' , take the maximum value of a prediction. (used for classifications)
                               'weighted', compute the weighted average of a prediction. A prediction is an array
                                containing the score of each label. The top score label predict will have the highest 
                                weight, the second scor the second highest weight, and so on. (used for regression)
    """
    predict_x = []
    predict_y = []
    X = np.linspace(-size/10, size/10, N)

    if read_out == 'top':
        for i in range(len(predictions_x)):
            predict_x.append(X[np.argmax(predictions_x[i])])
            predict_y.append(X[np.argmax(predictions_y[i])])

    if read_out == 'weighted':
        for i in range(len(predictions_x)):
            predict_x.append(np.dot(predictions_x[i], X))
            predict_y.append(np.dot(predictions_y[i], X))
    
    distance = []

    for i in range(len(predict_x)):
        dist = mt.sqrt((abs(predict_x[i] - real_x[i]))**2 + (abs(predict_y[i] - real_y[i]))**2)
        distance.append(dist)

    dist_x = []
    dist_y = []

    for i in range(len(predict_x)):
        dist_x.append(abs(predict_x[i] - real_x[i]))
        dist_y.append(abs(predict_y[i] - real_y[i]))
    dist_x = np.asarray(dist_x)
    dist_y = np.asarray(dist_y)
    idx = (dist_x > 2) & (dist_y > 2) 
    plt.scatter(real_x[idx], real_y[idx])
    plt.show()
    distance = np.asarray(distance)
    plt.subplot(1, 3, 1)
    plt.hist(np.minimum(10, dist_x))
    plt.xlabel('Distance (abs) between x_coor_net and x_coor_real')
    plt.ylabel('Number of point')
    plt.title('CNN for x_coord')
    plt.subplot(1, 3, 2)
    plt.xlabel('Distance (abs) between y_coor_net and y_coor_real')
    plt.ylabel('Number of point')
    plt.title('CNN for y_coord')
    plt.hist(np.minimum(10, dist_y))
    plt.subplot(1, 3, 3)
    plt.xlabel('Distance (abs) between point_net and point_real')
    plt.ylabel('Number of point')
    plt.title('Distance between prevision and real')
    plt.hist(np.minimum(10, distance))
    plt.show() 
    return distance



'''
train_images, train_labels, train_rays = generateArtificialSet(128, 128, 5, 10, 10000)
test_images, test_labels, test_rays = generateArtificialSet(128, 128, 2, 20, 3000)



train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = np.reshape(train_images, (train_images.shape[0], 128, 128, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 128, 128, 1))

x_labels = util.gaussWeights(train_labels[:,0], 20, 128, 2)
y_labels = util.gaussWeights(train_labels[:,1], 20, 128, 2)
x_test = util.gaussWeights(test_labels[:,0], 20, 128, 2)
y_test = util.gaussWeights(test_labels[:,1], 20, 128, 2)

plt.plot(x_labels[0])
plt.show()


model_1 = createModel()
model_2 = createModel()

trainModel(model_1, train_images, x_labels, test_images, x_test, n_epoch = 10, filename = 'esperimento_artificiale_x.h5')
trainModel(model_2, train_images, y_labels, test_images, y_test, n_epoch = 10, filename = 'esperimento_artificiale_y.h5')

prediction_x = predictImages(model_1, test_images)
prediction_y = predictImages(model_2, test_images)

histogram2(prediction_x, prediction_y, test_labels[:,0], test_labels[:,1])
'''

model_1 = createModel()
model_2 = createModel()



train_images, train_labels, train_rays = generateArtificialSet(128, 128, 2, 21, 10000)
test_images, test_labels, test_rays = generateArtificialSet(128, 128, 2, 21, 3000)
train_images, test_images = train_images / 255.0, test_images / 255.0

train_images = np.reshape(train_images, (train_images.shape[0], 128, 128, 1))
test_images = np.reshape(test_images, (test_images.shape[0], 128, 128, 1))

x_labels = util.gaussWeights(train_labels[:,0], N, 128, 2)
y_labels = util.gaussWeights(train_labels[:,1], N, 128, 2)
x_test = util.gaussWeights(test_labels[:,0], N, 128, 2)
y_test = util.gaussWeights(test_labels[:,1], N, 128, 2)

trainModel(model_1, train_images, x_labels, test_images, x_test, n_epoch = 10, filename = 'esperimento_artificiale2_x.h5')
trainModel(model_2, train_images, y_labels, test_images, y_test, n_epoch = 10, filename = 'esperimento_artificiale2_y.h5')

prediction_x = predictImages(model_1, test_images)
prediction_y = predictImages(model_2, test_images)

distance = histogram2(prediction_x, prediction_y, test_labels[:,0], test_labels[:,1], 'weighted')

l = list([test_rays[i], distance[i]]  for i in range(len(distance)))
l = sorted(l, key=itemgetter(0))
l = np.asarray(l)
X = np.linspace(2,20,num=19)
a = []

for i in X:
    idx = l[:,0] == i**2
    a.append(np.max(l[idx, 1]))
plt.plot(X, a)
plt.ylabel("Differences between real and prevision")
plt.xlabel("Pupil Radius")
plt.show()

