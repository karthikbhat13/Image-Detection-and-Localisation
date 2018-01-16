import cv2
import numpy as np
import os
import glob
from random import shuffle
from tqdm import tqdm
from xml.dom import minidom
from keras.models import load_model
import matplotlib.pyplot as plt
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

PATH =  "../VOCdevkit/VOC2010/Annotations"
TRAIN_DIR = 'Train'
TEST_DIR = 'Test'
IMG_SIZE = 50
LR = 1e-3
NoOfSubjects = 20
bboxes = 4


############################################################################################


RegressionModel = load_model("VOCdatset-0.001-4conv.model.Regression")
Classifymodel = load_model("VOCdatset-0.001-4conv.model.Classification")

#########################################################################

print("Parsing Annotations...")
allFiles = []
for root, dirs, filenames in os.walk(PATH):
    for f in filenames:
        if f.endswith('.xml'):
            allFiles.append(os.path.join(root, f))

#print(allFiles)

mainDic = {}


for f in allFiles:
    xmldoc = minidom.parse(f)
    width = xmldoc.getElementsByTagName('width')
    height = xmldoc.getElementsByTagName('height')

    name = xmldoc.getElementsByTagName('name')
    file_name = xmldoc.getElementsByTagName('filename')
    ymin = xmldoc.getElementsByTagName('ymin')
    ymax = xmldoc.getElementsByTagName('ymax')
    xmin = xmldoc.getElementsByTagName('xmin')
    xmax = xmldoc.getElementsByTagName('xmax')
    k = 0
    #print(len(name))
    dic = {}
    for i in name:
        j = i.firstChild.nodeValue
        dim = []
        dim.extend([xmin[k].firstChild.nodeValue, ymin[k].firstChild.nodeValue, xmax[k].firstChild.nodeValue, ymax[k].firstChild.nodeValue])

        dic[j] = dim
        mainDic[file_name[0].firstChild.nodeValue] = dic
        k+=1
        #print(j)
#print(mainDic)
#######################################################################################
print("Prediction MODE")

Classes = np.load('Classes.npy')
testing_data = []
testing_data1 = []
test_names = []
for img in tqdm(os.listdir(TEST_DIR)):
	path = os.path.join(TEST_DIR,img)
	imgs = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
	imgs1 = cv2.resize(cv2.imread(path),(IMG_SIZE,IMG_SIZE))
	test_names.append(img)
	testing_data.append(np.array(imgs))
	#test_names.append(img)
	#testing_data1.append(np.array(imgs1))

fig = plt.figure()
k = 1
c_label = {}
RESIZE_CONSTANT = 10
y_pred = []
for i in range(len(testing_data)):
    print(test_names[i])
    print(mainDic[test_names[i]])
    predictor = testing_data[i].reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)
    Regresspredict = RegressionModel.predict(predictor)
    print(Regresspredict)
    

    Classifypredict = Classes[np.argmax(Classifymodel.predict(testing_data[i].reshape(-1 ,IMG_SIZE, IMG_SIZE, 1)))]
    img = cv2.imread("Test/" + test_names[i])
    img = cv2.rectangle(img, (int(Regresspredict[0][0] - RESIZE_CONSTANT) , int(Regresspredict[0][1] - RESIZE_CONSTANT)), (int(Regresspredict[0][2] + RESIZE_CONSTANT) ,int(Regresspredict[0][3] + RESIZE_CONSTANT )), (255, 0, 0), 2)
    y = fig.add_subplot(2,2,k)
    y.imshow(img)
    plt.title(Classifypredict)
    k+=1
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
print(RegressionModel.summary())
print(Classifymodel.summary())
################################################################################################
