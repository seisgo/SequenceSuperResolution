'''
Author: Zhenyu Yuan
Email: zhenyuyuan@outlook.com
Brief: model defination for sequence super resolution model
Refer: https://github.com/leekh7411/Fast-Audio-Super-Resolution/blob/master/model.py
'''

import tensorflow as tf
import tensorflow.keras as keras
import math

## Define function for upsampling
def SubPixel1D(input_shape, r=2, color=False):
    def _phase_shift(I, r):
        X = tf.transpose(I)
        X = tf.batch_to_space_nd(X, [r], [[0,0]])
        X = tf.transpose(X)
        return X

    def subpixel_shape(input_shape):
        dims = [input_shape[0], input_shape[1]*r, int(input_shape[2]/r)]
        output_shape = tuple(dims)
        return output_shape
    
    def subpixel(x):
        # only single channel
        x_upsampled = _phase_shift(x, r)
        return x_upsampled

    return keras.layers.Lambda(subpixel, output_shape=subpixel_shape)

def ssr_model(summary=True, ns=256):
    # judge whether the input shape is an exponent of 2
    expV = math.log2(ns)
    if math.ceil(expV) != math.floor(expV):
        raise ValueError
    expV = int(expV)-2

    print('load base model..')
    x = keras.layers.Input((ns,1))
    main_input = x

    # Downsampling layers
    dList = list()
    for i in range(expV):
        fSize = 32
        if i==0:
            fSize = 16
        np = expV-i-2
        if np<2:
            np = 2
        kSize = pow(2,np)
        x = keras.layers.Conv1D(padding='same', kernel_initializer='Orthogonal', filters=fSize,
                                kernel_size=kSize, activation=None, strides=2)(x)
        x = keras.layers.LeakyReLU(0.2)(x)
        dList.append(x)
            
    # Bottleneck layer 
    x = keras.layers.Conv1D(padding='same', kernel_initializer='Orthogonal', filters=32,
                            kernel_size=4, activation=None, strides=2)(x)
    x = keras.layers.LeakyReLU(0.2)(x)

    # Upsampling layers
    for i in range(expV):
        fSize = 64
        if i == expV-1:
            fSize = 32
        np = i-1
        if np<2:
            np = 2
        kSize = pow(2, np)
        x = keras.layers.Conv1D(padding='same', kernel_initializer='Orthogonal',filters=fSize,
                                kernel_size=kSize, activation=None)(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(rate=0.5)(x)
        x = SubPixel1D(x.shape, r=2, color=False)(x)
        x = keras.layers.concatenate([x, dList[-1-i]])
    
    # SubPixel-1D Final
    x = keras.layers.Conv1D(padding='same', kernel_initializer='he_normal',filters=2,
                            kernel_size=16, activation=None)(x)     
    x = SubPixel1D(x.shape, r=2, color=False)(x)
    output = keras.layers.add([x, main_input])
    model  = keras.models.Model(main_input,output)
    
    if summary: 
        model.summary()       
        
    return model