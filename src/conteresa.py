'''
Created on 07/11/2019

@author: ap
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import util 
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import cv2
import pyCompare
import random as rd
import lsr.util
#DEFINE NAME OF .npy FILE
IMAGE_FILE = 'Train_Set_valid\\Training_64.npy'
LABELS_x_FILE = 'Train_Set_valid\\Training_64_label_x.npy'
LABELS_y_FILE = 'Train_Set_valid\\Training_64_label_y.npy'
#LOADING IMAGE SET
(train_images, test_images) = util.loadTrainingSet(IMAGE_FILE)
#LOAD LABELS 
(train_x_labels, test_x_labels) = util.loadLabels(LABELS_x_FILE)
(train_y_labels, test_y_labels) = util.loadLabels(LABELS_y_FILE)
#train_labels = np.asarray(util.combineLabel(train_x_labels, train_y_labels))
#test_labels = np.asarray(util.combineLabel(test_x_labels, test_y_labels))
#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[0:-1]

#APPLY THE GAUSSIAN DISTRIBUTION FOR EACH LABEL (UTIL TO COMPUTE THE COSINE SIMILARITY)
N, size = 20, 64
train_x_labels = util.cosineSimilarity(train_x_labels, N, size)
train_y_labels = util.cosineSimilarity(train_y_labels, N, size)
test_x_labels = util.cosineSimilarity(test_x_labels, N, size)
test_y_labels = util.cosineSimilarity(test_y_labels, N, size)
'''
train_x_labels, test_x_labels = util.rescaleLabels(train_x_labels, 0, 1504, 0, 64), util.rescaleLabels(test_x_labels, 0, 1504, 0, 64)
train_y_labels, test_y_labels = util.rescaleLabels(train_y_labels, 0, 1496, 0, 64), util.rescaleLabels(test_y_labels, 0, 1496, 0, 64)
'''
'''
#Normalize labels
train_x_labels = ((train_x_labels + 32).astype(int))/64
train_y_labels = ((train_y_labels - 32)*-1).astype(int)
test_x_labels =  ((test_x_labels + 32).astype(int))/64
test_y_labels =  ((test_y_labels - 32)*-1).astype(int)
#train_labels = np.asarray(util.combineLabel(train_x_labels, train_y_labels))
#test_labels = np.asarray(util.combineLabel(test_x_labels, test_y_labels))
'''
def createModel(image_shape = (64, 64, 1), kernel_size = (3, 3)):
    """
    Create a model of Convolutional Neural Network, with 4 layers of convolution.

    image_shape (tuple) --> tuple containing the dimensions of the images. Tensorflow expect RGB images, 
                            so the tuple must be (n,m,3). ---Default value is (64, 64, 3)---
    kernel_size (tuple) --> kernel size for convolution.  ---Default value is (3,3)---
    """
    
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size, activation = 'relu', padding='same', input_shape= image_shape))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64))
    model.add(layers.Dense(1))

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

    print(model.summary())
    return model




def trainModel(model, train_images, train_labels, n_epoch= 10, filename='void'):
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
    history = model.fit(train_images, train_labels, epochs = n_epoch, validation_data=(test_images, test_x_labels))
    plt.subplot(2,1,1)
    plt.plot(history.history['mse'], label='mse')
    plt.plot(history.history['val_mse'], label = 'val_mse')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.subplot(2,1,2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
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

def histogram(predictions, read_out = 'classification'):
    """
    Return the histogram of the distance between prevision and real value.
    predictions (2D array) --> results of the previsions of a DNN.
    read_out (string)      --> define the read out method (default value 'classification'). 
                               'classification' , take the maximum value of a prediction. (used for classifications)
                               'weighted', compute the weighted average of a prediction. A prediction is an array
                                containing the score of each label. The top score label predict will have the highest 
                                weight, the second scor the second highest weight, and so on. (used for regression)
    """
    
    if read_out == 'classification':
        predict = []
        for i in range(len(predictions)):
            predict.append(abs(predictions[i] - test_x_labels[i]*64)) 
        plt.hist(predict)
        plt.xlabel('Distance (abs) between coord_net and coord_real')
        plt.ylabel('Number of point')
        plt.title('Read out: classification')
        plt.show()

    if read_out == 'weighted':
        predict = []
        for i in range(len(predictions)):
            weight = 64
            mean = 0
            sum_weight = 0
            for j in range(len(predictions[i])):
                ma_x = np.argmax(predictions[i])
                mean += ma_x * weight
                predictions[i][ma_x] = 0
                sum_weight += weight
                weight -= 1   
            predict.append(abs(mean/sum_weight - test_x_labels[i]))
        plt.hist(predict)
        plt.xlabel('Distance (abs) between coord_net and coord_real')
        plt.ylabel('Number of point')
        plt.title('Read out: weighted')
        plt.show()
    

def histogram2(predictions_x, prediction_y, read_out = 'classification'):
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

    if read_out == 'classification':
        for i in range(len(predictions_x)):
            predict_x.append(np.argmax(predictions_x[i]))
            predict_y.append(np.argmax(predictions_y[i]))
    
        distance = []

        for i in range(len(predict_x)):
            dist = mt.sqrt((abs(predict_x[i] - test_x_labels[i]))**2 + (abs(predict_y[i] - test_y_labels[i]))**2)
            distance.append(int(dist))

        dist_x = []
        dist_y = []

        for i in range(len(predict_x)):
            dist_x.append(abs(predict_x[i] - test_x_labels[i]))
            dist_y.append(abs(predict_y[i] - test_y_labels[i]))

    test_x = np.asarray(test_x_labels)
    test_y = np.asarray(test_y_labels)
    dist_x = np.asarray(dist_x)
    dist_y = np.asarray(dist_y)
    idx = (dist_x > 2) & (dist_y > 2) 
    plt.scatter(test_x[idx], test_y[idx])
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


'''
train_images = np.reshape(train_images, (11064, 64, 64, 1))
test_images = np.reshape(test_images, (2767, 64, 64, 1))
model_1 = createModel()
trainModel(model_1, train_images, train_x_labels, n_epoch = 15, filename = 'esperimento_10_x.h5')
'''

'''
test_images = np.reshape(test_images, (2767, 64, 64, 1))
model = tf.keras.models.load_model('esperimento_10_x.h5')
prediction = predictImages(model, test_images)
prediction = np.asarray(np.transpose(prediction))
histogram((prediction[0]*64).astype(int))
print(prediction[0]*64, train_x_labels*64)
'''

'''
model_x = createModel()
model_y = createModel()



trainModel(model_x, train_images, train_x_labels, n_epoch = 15, filename = 'esperimento_10_x.h5')
trainModel(model_y, train_images, train_y_labels, n_epoch = 15, filename = 'esperimento_10_y.h5')

predictions_x = predictImages(model_x, test_images)
predictions_y = predictImages(model_y, test_images)

histogram2(predictions_x, predictions_y)
'''

'''
plt.scatter(test_x_labels, test_y_labels)

model_x = tf.keras.models.load_model('Modelli\\Esperimenti validi\\Esperimento 3\\esperimento_3_x.h5')
model_y = tf.keras.models.load_model('Modelli\\Esperimenti validi\\Esperimento 3\\esperimento_3_y.h5')

lsr.util.tic()
predictions_x = predictImages(model_x, test_images)
predictions_y = predictImages(model_y, test_images)
lsr.util.toc()
histogram2(predictions_x, predictions_y)


predict_x = [] 
predict_y = []

for i in range(len(predictions_x)):
    predict_x.append(np.argmax(predictions_x[i]))
    predict_y.append(np.argmax(predictions_y[i]))

distance = []

for i in range(len(predict_x)):
    dist = mt.sqrt((abs(predict_x[i] - test_x_labels[i]))**2 + (abs(predict_y[i] - test_y_labels[i]))**2)
    distance.append(int(dist))

#histogram2(predictions_x, predictions_y)
pyCompare.blandAltman(predict_x, test_x_labels)
util.bland_altman_plot(predict_y, test_y_labels)
predizione_x = util.rescaleLabels(predict_x,  0, 64, 0, 1504)
reale_x = util.rescaleLabels(test_x_labels,  0, 64, 0, 1504)
plt.scatter(predizione_x, reale_x, s=10, alpha=0.1)
util.regression(predizione_x, reale_x)

'''

'''
model_x = tf.keras.models.load_model('Modelli\\Esperimenti validi\\Esperimento 1\\esperimento_1_x.h5')

model_x.summary()
'''

'''
plt.imshow(train_images[0])
plt.show()
'''