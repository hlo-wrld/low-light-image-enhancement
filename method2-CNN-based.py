import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

img = cv.imread('/kaggle/input/low-light-proj/Train/high/102.png')  
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)

imgl = cv.imread('/kaggle/input/low-light-proj/Train/low/102.png')  
imgl = cv.cvtColor(imgl, cv.COLOR_BGR2RGB)
plt.imshow(imgl)

from math import log10, sqrt 
import cv2 
import numpy as np 

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    abs_er = np.abs(original - compressed)
    mae = np.mean(abs_er)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    print ("psnr: ",psnr,'mse: ',mse,'mae: ',mae)


original = cv2.imread("/kaggle/input/low-light-proj/Train/high/102.png") 
compressed = cv2.imread("/kaggle/input/low-light-proj/Train/low/102.png", 1) 
PSNR(original, compressed) 

def PreProcessData(imgpath_h,imgpath_l):
    X_=[]
    y_=[]
    for imageDir in os.listdir(imgpath_h):
        img = cv.imread(imgpath_h + imageDir)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_y = cv.resize(img,(500,500))
        y_.append(img_y)
    for imageDir in os.listdir(imgpath_l):
        img = cv.imread(imgpath_h + imageDir)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_y = cv.resize(img,(500,500))
        X_.append(img_y)
    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_
  
inp_h = '/kaggle/input/low-light-proj/Train/high/'
inp_l = '/kaggle/input/low-light-proj/Train/low/'
X_,y_ = PreProcessData(inp_h,inp_l)

K.clear_session()
def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5

Input_Sample = Input(shape=(500, 500,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)

Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
Model_Enhancer.summary()
from keras.utils.vis_utils import plot_model
plot_model(Model_Enhancer,to_file='model_.png',show_shapes=True, show_layer_names=True)
from IPython.display import Image
Image(retina=True, filename='model_.png')

def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,500,500,3)
        y_input = y[i].reshape(1,500,500,3)
        yield (X_input,y_input)
Model_Enhancer.fit_generator(GenerateInputs(X_,y_),epochs=5,verbose=1,steps_per_epoch=97,shuffle=True)

def ExtractTestInput(ImagePath):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_ = cv.resize(img,(500,500))
    img_ = img_.reshape(1,500,500,3)
    return img_

ImagePath='/kaggle/input/low-light-proj/Train/low/118.png'
image_for_test = ExtractTestInput(ImagePath)
Prediction = Model_Enhancer.predict(image_for_test)

Prediction = Prediction.reshape(500,500,3)
original.resize(500,500,3)
PSNR(original,Prediction)

import numpy as np
import cv2

# Assuming img_float is your float array
# Example float array in the range [0, 1]
img_float = Prediction  # Replace with your float image

# Scale the float values to the range [0, 255] if they are in the range [0, 1]
if img_float.max() <= 1.0:
    img_float = img_float * 255

# Convert to uint8
img_uint8 = img_float.astype(np.uint8)

# Convert to grayscale if the image has multiple channels
if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
    img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img_uint8  # Already a single channel image

# Save or display the image
cv2.imwrite('grayscale_image.jpg', img_gray)
plt.imshow(img_gray)

path = '/kaggle/input/low-light-proj/Train/low/14.png'
#image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = img_gray

# Compute the histogram
hist, bins = np.histogram(image.flatten(), 256, [0,256])

# Compute the CDF
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Mask all pixels with zero CDF value
cdf_m = np.ma.masked_equal(cdf, 0)

# Compute the histogram equalization lookup table
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# Apply the lookup table to get the equalized image
equalized_image = cdf[image]

# Plot the original and equalized images and their histograms
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_image, cmap='gray')

img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
plt.imshow(img_color)
img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
plt.imshow(img_color)
PSNR(original,img_color1)

img_colormap = cv2.applyColorMap(equalized_image, cv2.COLORMAP_JET)
plt.imshow(img_colormap)
PSNR(original,img_colormap)
