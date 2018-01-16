import cv2
import numpy as np
import os
import glob
from random import shuffle



TRAIN_DIR = 'VOCdevkit/VOC2010/JPEGImages'
#TEST_DIR = 'Test'
IMG_SIZE = 50
LR = 1e-3
NoOfSubjects = 20

MODEL_NAME = 'VOCdatset-{}-{}.model'.format(LR,'6conv')

def one_hot(i):
    a = np.zeros(NoOfSubjects)
    a[i] = 1
    return a

def create_train_data():
    training_data = []
    ClassFiles = glob.glob('VOCdevkit/VOC2010/ImageSets/Main/*_trainval.txt')
    #print(ClassFiles)
    ClassNames = [i.split('_')[0] for i in ClassFiles]
    for i in range(len(ClassFiles)):
        with open(ClassFiles[i],"r") as f:
        	images = []
        	for line in f:
        		arr = line.split()
        		if(int(arr[1])==1):
        			images.append(arr[0])

        label = one_hot(i)
        for files in images:
            path = os.path.join(TRAIN_DIR,(files + '.jpg'))
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        print(ClassFiles[i])
    print("Total Training dataset: ",len(training_data))
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return(training_data)


train_data = create_train_data()
test_data = train_data[-4500:]
train_data = train_data[:-4500]

print("Images Used for training: ",len(train_data))
#train_data = np.load('train_data.npy')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = fully_connected(convnet, 64, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, NoOfSubjects, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir = 'log')

train = train_data[:-4500]
test = train_data[-4500:]

X = np.array([i[0] for i in train]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
    snapshot_step=500, show_metric=True, run_id = MODEL_NAME)

model.save(MODEL_NAME)
