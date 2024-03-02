import os
import numpy as np
import tensorflow as tf
import pysm3
import pysm3.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
from time import perf_counter
import math

def normalize_data(data):
    # Check for zero division
    if np.max(data) - np.min(data) == 0:
        # If the range is zero, avoid division by zero and return the original data
        return data
    else:
        # Perform normalization
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        return normalized_data

def bin2nbit(binary,n):
  while(len(binary)<n):
    binary='0'+binary
  return binary

nside=1024
frequency=[85,95,145,155,220,270]
fwhm=np.array([25.5,22.7,25.5,22.7,13,13])*np.pi/(180*60)
sensit=np.array([1.31,1.15,1.78,1.91,4.66,7.99])
def generate_data_tensors(nside,k):
    t1=perf_counter()
    sky= pysm3.Sky(nside=nside, preset_strings=["s1", "f1", "a1", "d1"])
    sky_cmb=pysm3.Sky(nside=nside,preset_strings=["c1"])
    mpo=np.zeros((nside,nside))
    mp=np.zeros((nside,nside,6))
    for r in range(6):
        cmb=sky_cmb.get_emission(frequency[r]*u.GHz)
        map = sky.get_emission(frequency[r] * u.GHz)
        map = map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(frequency[r]*u.GHz))
        cmb = cmb.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(frequency[r]*u.GHz))
        fmap=map[0]+cmb[0]
        mapg=hp.sphtfunc.smoothing(fmap,fwhm[r])
        fmapg=mapg+np.random.normal(scale=sensit[r],size=len(mapg))
        m=fmapg
        mo=cmb[0].value
        pixar='0'*20
        kb=bin2nbit(bin(k)[2:],4)
        pixar=np.array(list(kb+pixar))
        for i in range(nside):
            ib=np.array(list(bin2nbit(bin(i)[2:],math.log2(nside))))
            for index, value in np.ndenumerate(ib):
                pixar[4+2*index[0]]=value
            for j in range(128):
                jb=np.array(list(bin2nbit(bin(j)[2:],math.log2(nside))))
                for index, value in np.ndenumerate(jb):
                    pixar[5+2*index[0]]=value
                PixNested=int(''.join([str(score) for score in pixar]),2)
                PixRing=hp.nest2ring(nside,PixNested)
                mp[i,j,r]=m[PixRing]
                if (r==4): 
                    mpo[i,j]=mo[PixRing]
    return tf.convert_to_tensor(mp),tf.convert_to_tensor(mpo)


def create_datasets(num_train_samples):
    # Initialize empty lists to store training and validation data
    train_inputs, train_outputs = [], []

    # Generate training data
    for _ in range(num_train_samples):
        tensor_input, tensor_output = generate_data_tensors(nside,6)
        train_inputs.append(tensor_input)
        train_outputs.append(tensor_output)

    # Generate validation data

    # Convert lists to NumPy arrays
    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)
    train_inputs = normalize_data(train_inputs).astype(np.float32)
    train_outputs = normalize_data(train_outputs).astype(np.float32)

    # Convert NumPy arrays to TensorFlow tensors
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs))

    return train_dataset

@tf.keras.utils.register_keras_serializable()
def los(y_true,y_pred):
    r=y_pred-y_true
    losss=tf.math.reduce_euclidean_norm(r)
    return losss*losss

@tf.keras.utils.register_keras_serializable()
def peak_signal_noise_ratio(y_true, y_pred):
    g=tf.image.psnr(y_pred, y_true, max_val=1.0)/100
    h=1-g
    return h

@tf.keras.utils.register_keras_serializable()
def structural_similarity(y_true, y_pred):
    l = tf.image.ssim(y_pred, y_true, max_val=1.0)*100
    return l

@tf.keras.utils.register_keras_serializable()
def unet(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # Encoder
    conv1 = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Conv2D(16, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    pool3 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(conv3)

    # Bottleneck
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.Conv2D(128, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = tf.keras.layers.BatchNormalization()(conv4)

    # Decoder
    up5 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_initializer='he_normal')(up5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.BatchNormalization()(conv5)

    up6 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(up6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)
    conv6 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = tf.keras.layers.BatchNormalization()(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(up7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)
    conv7 = tf.keras.layers.Conv2D(32, 3, activation='linear', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = tf.keras.layers.BatchNormalization()(conv7)

    # Output layer
    outputs = tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same')(conv7)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_loop(num_epochs, num_train_samples, num_val_samples, nside, model_save_path='best_model.h5'):
    train_dataset = create_datasets(num_train_samples)
    model = unet((nside,nside,6))

    # Initialize some variables for tracking the best model
    best_loss = float('inf')
    best_epoch = 0

    # Define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(num_epochs):
        # Training step
        for inputs, targets in train_dataset:
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = los(targets, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Validation step
        val_loss = 0
        for _ in range(num_val_samples):
            val_inputs, val_targets = generate_data_tensors(nside,6)
            val_predictions = model.predict(np.expand_dims(val_inputs, axis=0))
            val_loss += los(val_targets, val_predictions)

        val_loss /= num_val_samples

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            model.save(model_save_path)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss.numpy()}")

    print(f"Best model saved at epoch {best_epoch + 1}, with validation loss: {best_loss.numpy()}")