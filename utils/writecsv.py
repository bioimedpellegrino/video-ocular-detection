'''
Created on 07/11/2019

@author: ap
'''

import os
import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lsr import store
import numpy as np
import matplotlib.pyplot as plt
import random


#DEFINE DIRECTORY PATH
DIR_PATH   = os.path.join(os.path.dirname(__file__), 'Training')
#DEFINE FILE NAME
FILENAME = os.path.join(DIR_PATH, 'Data' + os.path.sep + 'Training.dat')
#DEFINE IMAGES DIRECTORY NAME
IMG_DIR = tuple('subject_' + str(i) + '.dat' for i in (2, 3, 6, 7, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22))

def createArrayFromFile(filename,start_col=0, end_col=0):
    """
    Read a file .dat and save it in a numpy array.
    If start_col and/or end_col are != 0, then a sub-array from start_col to end_col-1 is returned.

    filename : Name of the file to scan (string);
    start_col : start column index (int);
    end_col : end column index (int);
    """
    if end_col == 0:
        return np.vstack(tuple(store.loadAsciiArray(filename)))[:,start_col:]
    return np.vstack(tuple(store.loadAsciiArray(filename)))[:,start_col:end_col]


def getInfoForTraining(data, idx):
    """
    Return a tuple of an image from a row present in dataset by using an index. 
    The tuple contains: (PATH OF THE IMAGE, X_COORD, Y_COORD, WIDTH, HEIGHT)
    data: dataset (np.array);
    idx: index of th row (int);
    """
    #GET THE ROW FROM THE DATASET
    dd = data[idx]
    #IF ID ARE MORE THAN 1000, SO IS AN ARTIFICIAL IMAGES (id_subject_artificial = id_subject + 1000)
    if dd[0] > 999:
        #SO IS NECESSARY TO ELIMINATE THE MARK FIRST, IN ORDER TO KNOW THE REAL SUBJECT ID
        dd[0] -= 1000
    #GET THE LIST OF THE FILE FROM THE DIRECTORY OF THE SUBJECT'S IMAGE 
    images_list = os.listdir(os.path.join(DIR_PATH, 'Images' + os.path.sep + IMG_DIR[int(dd[0])]))
    #GET IMAGE ID 
    image_name = images_list[int(dd[1])]
    #RETURN IMAGE PATH, X_COORD, Y_COORD, WIDTH, HEIGHT
    return (os.path.join(DIR_PATH, 'Images' + os.path.sep + IMG_DIR[int(dd[0])] + os.path.sep + image_name), int(dd[2]), int(dd[3]), int(dd[-2]), int(dd[-1]))


def createCSV(data, filename="path_coor_viewport.csv"):
    """
    Creaate a .csv file with: image_path, x_coord, y_coord, width, height
    filename: name of the .csv file (string)
    data: dataset (np.array)
    """
    filename = "File CSV\\" + filename
    f = open(filename, "w")
    f.write("PATH, X_COORD, Y_COORD, WIDTH, HEIGHT" + " \n")
    for i in range(len(data)):
        r = getInfoForTraining(data, i)
        f.write(r[0] + ",")
        f.write(str(r[1]) + ",")
        f.write(str(r[2]) + ",")
        f.write(str(r[-2]) + ",")
        f.write(str(r[-1]) + " \n")
    f.close()




if __name__ == '__main__':
    data = createArrayFromFile(FILENAME)
    createCSV(data)





