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

from skimage import data
from skimage.color import rgb2gray

#DEFINE NAME OF .npy FILE
IMAGE_FILE = 'Train_Set_valid\\Training_128.npy'
LABELS_x_FILE = 'Train_Set_valid\\Train_label_128_x.npy'
LABELS_y_FILE = 'Train_Set_valid\\Train_label_128_y.npy'
#LOADING IMAGE SET
(train_images, test_images) = util.loadTrainingSet(IMAGE_FILE)
#LOAD LABELS 
(train_x_labels, test_x_labels) = util.loadLabels(LABELS_x_FILE)
(train_y_labels, test_y_labels) = util.loadLabels(LABELS_y_FILE)
test_x = test_x_labels
test_y = test_y_labels

#Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[1:]
#APPLY THE GAUSSIAN DISTRIBUTION FOR EACH POINT 
N, size, sigma = 20, 128, 2
train_x_lab = util.cosineSimilarity(train_x_labels, N, size, sigma)
train_y_lab = util.cosineSimilarity(train_y_labels, N, size, sigma)
test_x_lab = util.cosineSimilarity(test_x_labels, N, size, sigma)
test_y_lab = util.cosineSimilarity(test_y_labels, N, size, sigma)


def createModel(image_shape = (size, size, 1), kernel_size = (5, 5)):
    """
    Create a model of Convolutional Neural Network, with 4 layers of convolution.

    image_shape (tuple) --> tuple containing the dimensions of the images. Tensorflow expect RGB images, 
                            so the tuple must be (n,m,3). ---Default value is (64, 64, 3)---
    kernel_size (tuple) --> kernel size for convolution.  ---Default value is (3,3)---
    """
    
    model = models.Sequential()
    model.add(layers.Conv2D(8, kernel_size, activation = 'relu', padding='same', input_shape= image_shape))
    model.add(layers.MaxPool2D(2,2))
    model.add(layers.Flatten())
    model.add(layers.Dense(N, activation='softmax'))

    model.compile(optimizer='adam',
              loss='cosine_similarity',
              metrics=['cosine_similarity'])

    print(model.summary())
    return model




def trainModel(model, train_images, train_labels, val_labels,  n_epoch= 10, filename='void'):
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
            predict.append(abs(predictions[i] - test_y[i])) 
        plt.hist(predict)
        plt.xlabel('Distance (abs) between coord_net and coord_real')
        plt.ylabel('Number of point')
        plt.title('Read out: classification')
        plt.show()
    


def trainModel2(model, train_images, train_labels, n_training, sigma, n_neurons, n_epoch, filename):
    train_labels_gauss = util.cosineSimilarity(train_labels, n_neurons, train_images.shape[1], sigma)
    history = model.fit(train_images, train_labels_gauss, epochs = n_epoch, validation_data=(test_images, test_x_labels))

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

    for i in range(n_training):
        sigma = sigma / 2
        model = tf.keras.models.load_model(filename)
        train_labels_gauss = util.cosineSimilarity(train_labels, n_neurons, train_images.shape[1], sigma)
        model.fit(train_images, train_labels_gauss, epochs = n_epoch, validation_data=(test_images, test_x_labels))
        model.save(filename)




def histogram2(predictions_x, predictions_y, read_out = 'top'):
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
        dist = mt.sqrt((abs(predict_x[i] - test_x[i]))**2 + (abs(predict_y[i] - test_y[i]))**2)
        distance.append(dist)

    dist_x = []
    dist_y = []

    for i in range(len(predict_x)):
        dist_x.append(abs(predict_x[i] - test_x[i]))
        dist_y.append(abs(predict_y[i] - test_y[i]))
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
    return predict_x, predict_y


plt.plot(train_x_lab[0])
plt.show()
train_images = np.reshape(train_images, (11064, size, size, 1))
test_images = np.reshape(test_images, (2767, size, size, 1))
model_1 = createModel()
model_2 = createModel()
trainModel(model_1, train_images, train_x_lab, test_x_lab, n_epoch = 10, filename = 'esperimento_10_x.h5')
trainModel(model_2, train_images, train_y_lab, test_y_lab, n_epoch = 10, filename = 'esperimento_10_y.h5')
prediction_x = predictImages(model_1, test_images)
prediction_y = predictImages(model_2, test_images)
histogram2(prediction_x, prediction_y)

'''
X = np.linspace(int(-size/10), int(size/10), N)

prediction = predictImages(model_1, test_images)
predict = np.zeros(len(test_images))
for i in range(len(prediction)):
    #predict[i] = np.dot(prediction[i],X)
    predict[i] = X[np.argmax(prediction[i])]
histogram(predict)
'''

'''
model_x = tf.keras.models.load_model('esperimento_10_x.h5')
model_y = tf.keras.models.load_model('esperimento_10_y.h5')
test_images = np.reshape(test_images, (2767, size, size, 1))
prediction_x = predictImages(model_x, test_images)
prediction_y = predictImages(model_y, test_images)
histogram2(prediction_x, prediction_y)
'''
'''
train_images = np.reshape(train_images, (11064, 64, 64, 1))
test_images = np.reshape(test_images, (2767, 64, 64, 1))
model_1 = createModel()
trainModel2(model_1, train_images, train_x_labels, 5, 3, N, 5, filename = 'esperimento_10_x.h5')

X = np.linspace(-64/10, 64/10, N)

prediction = predictImages(model_1, test_images)
predict = np.zeros(len(test_images))
for i in range(len(prediction)):
    predict[i] = np.dot(prediction[i],X)
histogram(predict)
'''
'''
plt.plot(train_x_labels[0])
plt.show()
test_images = np.reshape(test_images, (2767, 64, 64, 1))
model = tf.keras.models.load_model('esperimento_10_x.h5')
prediction = predictImages(model, test_images)
X = np.linspace(-64/10, 64/10, N)

for i in prediction:
    print(np.dot(i,X))
'''
'''
model_x = tf.keras.models.load_model('Esperimenti\\Esperimento 256 valido\\esperimento_1_256_x.h5')
model_y = tf.keras.models.load_model('Esperimenti\\Esperimento 256 valido\\esperimento_1_256_y.h5')
model_x.summary()
test_images = np.reshape(test_images, (2767, size, size, 1))

lsr.util.tic()
prediction_x = predictImages(model_x, test_images)
prediction_y = predictImages(model_y, test_images)
lsr.util.toc()
#histogram2(prediction_x, prediction_y, 'weighted')
prev_x, prev_y = histogram2(prediction_x, prediction_y)
prev_x = np.asarray(prev_x) 
prev_y = np.asarray(prev_y)

test_images = np.squeeze(test_images)
print("Real - X: ", test_x[0] + 128, "Y: ", 128 - test_y[0], "Prevision - X:", 128 + prev_x[0], "Y:", 128  - prev_y[0])
'''
'''
for i in range(len(test_images[0:10])):
    plt.subplot(2,1,1)
    plt.imshow(test_images[i], cmap='gray')
    plt.plot(64 + test_x[i], 64 - test_y[i], 'r+', color='red')
    plt.title('Real center')
    plt.subplot(2,1,2)
    plt.imshow(test_images[i], cmap='gray')
    plt.plot(64 + prev_x[i], 64 - prev_y[i], 'r+', color='red')
    plt.title('Prevision')
    plt.show()
'''



'''
test_x = plt.imread("test_x.png")
test_x = rgb2gray(test_x).reshape(256,256,1)
test = []
test.append(test_x)
test.append(test_x)
test = np.asarray(test)

model_x = tf.keras.models.load_model('Esperimenti\\Esperimento 256 valido\\esperimento_1_256_x.h5')
model_y = tf.keras.models.load_model('Esperimenti\\Esperimento 256 valido\\esperimento_1_256_y.h5')

prediction_x = predictImages(model_x, test)
prediction_y = predictImages(model_y, test)

X = np.linspace(-256/10, 256/10, 40)
prev_x = np.max(X[np.argmax(prediction_x[0])])
prev_y = np.max(X[np.argmax(prediction_x[0])])

plt.imshow(np.squeeze(test[0]), cmap='gray')
plt.plot(prev_x + 128, 128 - prev_y, 'r+', color='red')
plt.title('Prevision')
plt.show()
'''
''
'''
model_x = tf.keras.models.load_model('Esperimenti\\Esperimento 128_2 valid\\esperimento_10_x.h5')
model_y = tf.keras.models.load_model('Esperimenti\\Esperimento 128_2 valid\\esperimento_10_y.h5')


test_images = np.reshape(test_images, (2767, size, size, 1))

prediction_x = predictImages(model_x, test_images)
prediction_y = predictImages(model_y, test_images)

prev_x, prev_y = histogram2(prediction_x, prediction_y)
prev_x = np.asarray(prev_x) 
prev_y = np.asarray(prev_y)

center = util.readFileCSV('centri.dat')

x = [70.94092746747435, 76.39084735, 59.52026909, 74.61756337, 51.26416289, 56.40750334, 64.95174431, 66.16765462, 64.25892607, 73.74599711]
y = [62.5004819751601, 63.02022386, 72.56753441, 69.048021,   68.50720201, 61.61307099, 68.27494021, 70.1651524,  66.55560093, 76.15742459]

test_images = np.squeeze(test_images)

for i in range(len(test_images[0:10])):
    plt.subplot(2,1,1)
    plt.imshow(test_images[i], cmap='gray')
    plt.plot(x[i], y[i], 'r+', color='red')
    plt.title('Real center')
    plt.subplot(2,1,2)
    plt.imshow(test_images[i], cmap='gray')
    plt.plot(64 + prev_x[i], 64 - prev_y[i], 'r+', color='red')
    plt.title('Prevision')
    plt.show()
'''

