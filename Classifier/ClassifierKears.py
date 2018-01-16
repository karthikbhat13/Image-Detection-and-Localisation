import cv2
import numpy as np
import os
import glob
from random import shuffle
from tqdm import tqdm
from xml.dom import minidom
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
import matplotlib.pyplot as plt



config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)

PATH =  "../VOCdevkit/VOC2010/Annotations"
TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
IMG_SIZE = 50
LR = 1e-3
NoOfSubjects = 20
bboxes = 4
MODEL_NAME = 'VOCdatset-{}-{}.model.Classification'.format(LR,'4conv')


######################################################################################
print("Creating Training Data...")
def one_hot(i):
    a = np.zeros(NoOfSubjects)
    a[i] = 1
    return a
i = 0
Classes = []
training_data = []
for folder in tqdm(os.listdir(TRAIN_DIR)):
    path_main = os.path.join(TRAIN_DIR,folder)
    Classes.append(folder)
    for img in tqdm(os.listdir(path_main)):
        label = one_hot(i)
        path = os.path.join(path_main,img)
        img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    i += 1
shuffle(training_data)
np.save('train_data.npy', training_data)
print(training_data[0])
print("Total Training Data Available: ", len(training_data))
np.save('Classes.npy',Classes)


#######################################################################################
print("Setting up CNN layers")

model = Sequential()

model.add(Conv2D(64, 3, 3, border_mode='same', input_shape = (IMG_SIZE, IMG_SIZE,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

#model.add(Conv2D(128, (2, 2), activation = "relu"))
#model.add(MaxPooling2D(pool_size = (2,2)))

#model.add(Conv2D(64, (2, 2 ), activation = "relu"))
#model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense((128), activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(NoOfSubjects, activation = "softmax"))
sgd = SGD(lr=LR, momentum=0.9, decay = LR, nesterov = False)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

train = training_data[:-5000]
test = training_data[-5000:]

X = np.array([np.array(i[0]) for i in train]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]
Y = np.array(Y)
test_x = np.array([np.array(i[0]) for i in test]).reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]
test_y = np.array(test_y)
#test_y = keras.utils.to_categorical(test_y)
model.fit(np.array(X), np.array(Y), validation_data = (test_x, test_y),epochs=100, batch_size = 1024)
model.save(MODEL_NAME)
#model.load_weights(MODEL_NAME)

print(model.summary())

######################################################################################
print("Prediction MODE")

Classes = np.load('Classes.npy')
testing_data = []
test_names = []
for img in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR,img)
    imgs = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
    test_names.append(img)
    testing_data.append(np.array(imgs))

fig = plt.figure()
k = 1
for i in range(len(testing_data)):
    print(test_names[i])
    prediction = np.argmax(model.predict(testing_data[i].reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)))
    print(Classes[prediction])
    img = cv2.imread("Test/" + test_names[i])
    y = fig.add_subplot(4,2,k)
    y.imshow(img)
    plt.title(Classes[prediction])
    k+=1
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
#####################################################################################
