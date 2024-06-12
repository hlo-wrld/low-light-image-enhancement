import os
import cv2
import numpy as np
from math import log10, sqrt 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import add, Conv2D, Input
from keras.models import Model

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    abs_er = np.abs(original - compressed)
    mae = np.mean(abs_er)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    print("PSNR:", psnr, 'MSE:', mse, 'MAE:', mae)
    return psnr

def PreProcessData(high_light_dir, low_light_dir):
    X_ = []
    y_ = []
    for imageDir in os.listdir(high_light_dir):
        img_h = cv2.imread(os.path.join(high_light_dir, imageDir))
        img_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB)
        img_h_resized = cv2.resize(img_h, (500, 500))
        y_.append(img_h_resized)
        
        img_l = cv2.imread(os.path.join(low_light_dir, imageDir))
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        img_l_resized = cv2.resize(img_l, (500, 500))
        X_.append(img_l_resized)
    
    return np.array(X_), np.array(y_)

def InstantiateModel(input_shape):
    input_layer = Input(shape=input_shape)
    
    model_1 = Conv2D(16, (3, 3), activation='relu', padding='same', strides=1)(input_layer)
    # Add more layers as needed
    
    output_layer = Conv2D(3, (3, 3), activation='relu', padding='same', strides=1)(model_1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

def GenerateInputs(X, y, batch_size=32):
    data_generator = ImageDataGenerator()
    data_generator.fit(X)
    data_flow = data_generator.flow(X, y, batch_size=batch_size, shuffle=True)
    return data_flow

def train_model(Model_Enhancer, X, y, epochs=5, batch_size=32):
    data_flow = GenerateInputs(X, y, batch_size=batch_size)
    Model_Enhancer.fit(data_flow, epochs=epochs, steps_per_epoch=len(X) // batch_size, verbose=1)

def test_model(Model_Enhancer, img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (500, 500))
    img_input = np.expand_dims(img_resized, axis=0)
    
    enhanced_img = Model_Enhancer.predict(img_input)
    enhanced_img = np.squeeze(enhanced_img)
    
    return enhanced_img

if __name__ == "__main__":
    # Define paths to high and low light images
    high_light_dir = '/kaggle/input/low-light-proj/Train/high/'
    low_light_dir = '/kaggle/input/low-light-proj/Train/low/'

    # Preprocess data
    X_, y_ = PreProcessData(high_light_dir, low_light_dir)

    # Instantiate model
    Model_Enhancer = InstantiateModel(input_shape=(500, 500, 3))

    # Compile model
    Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')

    # Train the model
    train_model(Model_Enhancer, X_, y_, epochs=5, batch_size=32)

    # Test the model with a sample image
    img_path = '/kaggle/input/low-light-proj/Train/low/118.png'
    enhanced_img = test_model(Model_Enhancer, img_path)

    # Display PSNR between original and enhanced images
    original = cv2.imread("/kaggle/input/low-light-proj/Train/high/102.png")
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    PSNR(original, enhanced_img)
