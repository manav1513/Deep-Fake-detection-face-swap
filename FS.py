import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


IMG_SIZE = 128  
EPOCHS = 50
BATCH_SIZE = 32

image_path_A = 'WhatsApp Image 2024-10-09 at 3.17.15 PM.jpeg'
image_path_B = 'WhatsApp Image 2024-10-09 at 3.17.15 PM.jpeg'


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to load image at {image_path}")
        return None 
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img / 255.0 
    return img


data_A = load_image(image_path_A)
data_B = load_image(image_path_B)

assert data_A.shape == (IMG_SIZE, IMG_SIZE, 3), "Image A should be of shape (128, 128, 3)"
assert data_B.shape == (IMG_SIZE, IMG_SIZE, 3), "Image B should be of shape (128, 128, 3)"

data_A = np.expand_dims(data_A, axis=0)  
data_B = np.expand_dims(data_B, axis=0)  

input_A = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)) 
input_B = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)) 


def create_encoder(input_layer):
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

def create_decoder(encoded):
    x = layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)
    return decoded


encoder_A = create_encoder(input_A)
decoder_A = create_decoder(encoder_A)
autoencoder_A = Model(inputs=input_A, outputs=decoder_A)

encoder_B = create_encoder(input_B)
decoder_B = create_decoder(encoder_B)
autoencoder_B = Model(inputs=input_B, outputs=decoder_B)


autoencoder_A.compile(optimizer=Adam(), loss='mean_squared_error')
autoencoder_B.compile(optimizer=Adam(), loss='mean_squared_error')


autoencoder_A.summary()
autoencoder_B.summary()

autoencoder_A.fit(data_A, data_A, epochs=EPOCHS, batch_size=BATCH_SIZE)
autoencoder_B.fit(data_B, data_B, epochs=EPOCHS, batch_size=BATCH_SIZE)


swapped_face_A = autoencoder_B.predict(data_A) 


plt.imshow(swapped_face_A[0]) 
