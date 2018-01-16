import cv2
import numpy as np
import os
import glob
from random import shuffle
#from tqdm import tqdm

import keras
from keras import layers
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K
config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)
	


def one_hot(i):
    a = np.zeros(NoOfSubjects)
    a[i] = 1
    return a

def create_train_data(IMG_SIZE, TRAIN_DIR):
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
            #print(img.shape)
            training_data.append([np.array(img), np.array(label)])
        print(ClassFiles[i])
    print("Total Training dataset: ",len(training_data))
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return(training_data)



def create_model(train, test, IMG_SIZE, NoOfSubjects):
	model = Sequential()

	model.add(Conv2D(128, 3, 3, border_mode='same', input_shape = (IMG_SIZE, IMG_SIZE,1)))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.2))

	#model.add(Conv2D(64, (3, 3), activation = "relu"))
	#model.add(MaxPooling2D(pool_size = (2,2)))
	#model.add(Dropout(0.2))
	#model.add(Conv2D(128, (2, 2), activation = "relu"))
	#model.add(MaxPooling2D(pool_size = (2,2)))

	#model.add(Conv2D(64, (2, 2 ), activation = "relu"))
	#model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Flatten())
	model.add(Dense((128), activation = "relu"))
	model.add(Dropout(0.4))

	model.add(Dense(NoOfSubjects, activation = "softmax"))

	epochs = 10
	lrate = 0.01

	sgd = SGD(lr=lrate, momentum=0.9, decay = lrate/epochs, nesterov = False)

	model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
	X, Y, test_x, test_y = data(train, test, IMG_SIZE)
	model.fit(np.array(X), np.array(Y), validation_data = (test_x, test_y),epochs=epochs, batch_size = 100)

	print(model.summary())	
	
	return model





def data(train, test, IMG_SIZE):

	X = np.array([np.array(i[0]) for i in train]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
	Y = [i[1] for i in train]
	Y = np.array(Y)
	#Y = keras.utils.to_categorical(Y)

	test_x = np.array([np.array(i[0]) for i in test]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
	test_y = [i[1] for i in test]
	test_y = np.array(test_y)
	#test_y = keras.utils.to_categorical(test_y)

	return X, Y, test_x, test_y



def load_data():
	train_data = np.load('train_data.npy')
	#train_data = create_train_data()
	test_data = train_data[-2000:]
	train_data = train_data[:-2000]

	print("Images Used for training: ",len(train_data))
	#train_data = np.load('train_data.npy')

	train = train_data[:-4500]
	test = train_data[-4500:]

	return train_data, test_data, train, test
	

	#K.set_image_dim_ordering('th')


def blah():	
	
	
	
	TRAIN_DIR = 'VOCdevkit/VOC2010/JPEGImages'
	#TEST_DIR = 'Test'
	IMG_SIZE = 50
	LR = 1e-3
	NoOfSubjects = 20

	MODEL_NAME = "VOCdatset-0.001-6conv.model.highernodes1.index"

	train_data, test_data, train, test  = load_data()
	#X, Y, test_x, test_y = data(train, test)

	model = create_model(train, test, IMG_SIZE, NoOfSubjects)	
	
	tr = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	ty = np.array([i[1] for i in test_data])


	scores = model.evaluate(tr, ty, verbose = 0)
	#print(model.predict(tr))
	print(scores)
	model.save(MODEL_NAME)

blah()