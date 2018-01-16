import cv2
import numpy as np
import os
import glob
from random import shuffle
from tqdm import tqdm

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



def one_hot(i, NoOfSubjects):
    a = np.zeros(NoOfSubjects)
    a[i] = 1
    return a

def create_train_data(IMG_SIZE, TRAIN_DIR, subjects, NoOfSubjects):
    i = 0
    training_data = []
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        path_main = os.path.join(TRAIN_DIR,folder)
        subjects.append(folder)
        for img in tqdm(os.listdir(path_main)):
            label = one_hot(i, NoOfSubjects)
            path = os.path.join(path_main,img)
            img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        i += 1
    shuffle(training_data)
    np.save('subject.npy', np.array(subjects))
    print(training_data[0])
    print(len(training_data[0][0]))
    return training_data



def create_model(train, test, IMG_SIZE, NoOfSubjects):
	
	model = Sequential()

	model.add(Conv2D(32, 3, 3, border_mode='same', input_shape = (IMG_SIZE, IMG_SIZE,1)))
	model.add(Dropout(0.2))

	model.add(Conv2D(32, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))
	"""
	model.add(Conv2D(128, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))
	
	model.add(Conv2D(256, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(256, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(128, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(128, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	
	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(64, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(32, (3, 3), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Conv2D(32, (3, 3), activation = "relu"))
	model.add(MaxPooling2D(pool_size = (2,2)))
	"""
	model.add(Flatten())
	

	model.add(Dense((256), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Dense((128), activation = "relu"))
	model.add(Dropout(0.2))

	model.add(Dense((32), activation = "relu"))
	model.add(Dropout(0.2))

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



def load_data(IMG_SIZE, TRAIN_DIR, subjects, NoOfSubjects):
	#train_data = np.load('train_data.npy')

	train_data = create_train_data(IMG_SIZE, TRAIN_DIR, subjects, NoOfSubjects)
	
	np.save('train_data.npy', train_data)
	
	
	test_data = train_data[-2000:]
	train_data = train_data[:-2000]

	print("Images Used for training: ",len(train_data))
	#train_data = np.load('train_data.npy')

	train = train_data[:-2000]
	test = train_data[-2000:]

	return train_data, test_data, train, test


	#K.set_image_dim_ordering('th')


def blah():
	config = K.tf.ConfigProto()
	config.gpu_options.allow_growth = True
	session = K.tf.Session(config=config)



	TRAIN_DIR = 'Train'
	#TEST_DIR = 'Test'
	IMG_SIZE = 64
	LR = 1e-3
	NoOfSubjects = 20

	MODEL_NAME = 'VOCdatset-{}-{}.model.croppedImages1'.format(LR,'6conv')

	subjects = []
	train_data, test_data, train, test  = load_data(IMG_SIZE, TRAIN_DIR, subjects, NoOfSubjects)
	X, Y, test_x, test_y = data(train, test, IMG_SIZE)
	model = create_model(train, test, IMG_SIZE, NoOfSubjects)


	tr = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
	ty = np.array([i[1] for i in test_data])


	#scores = model.evaluate(tr, ty, verbose = 0)
	#print(model.predict(tr))
	#print(scores)
	model.save(MODEL_NAME)

blah()
