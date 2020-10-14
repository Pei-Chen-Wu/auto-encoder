# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 22:16:56 2020

@author: Nclab
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers, optimizers, datasets, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Activation
import matplotlib.pyplot as plt
from sklearn import manifold

mnist = datasets.mnist
(x0, y0), (x1, y1) = datasets.mnist.load_data()
train_x = x0.reshape(-1,784)
train_y = x0.reshape(-1,784)
test_x = x1.reshape(-1,784)
test_y = x1.reshape(-1,784)
"""
dt = []
tsne=manifold.TSNE(n_components=2)
intermed_tsne=tsne.fit_transform(train_x)
for i in range(10):
    dt.append(intermed_tsne[y1[-10000:]==i])
color =['c', 'b', 'g', 'r', 'orange', 'y', 'k', 'silver','pink','purple']
plt.figure(figsize=(12,12))
for i in range(10):
    plot = dt[i]
    plt.scatter(plot[:,0],plot[:,1],c=color[i],label = i)
    plt.legend(loc = 'best')
plt.show()
"""
input_x = Input([784,])

#encoder
enc_input = Input([784,])
x = Dense(1000,activation= 'relu')(input_x)
x = Dense(500,activation= 'relu')(x) 
x = Dense(250,activation= 'relu')(x) 
enc_output = Dense(100)(x) 
encoder = Model(input_x,enc_output)
#decoder
dec_input = Input([100,])
x = Dense(250,activation= 'relu')(dec_input) 
x = Dense(500,activation= 'relu')(x) 
x = Dense(1000,activation= 'relu')(x) 
x = Dense(784)(x) 
dec_output = x
decoder = Model(dec_input,dec_output)
#合并

code = encoder(input_x )
output = decoder(code)

model = Model(input_x,output)

optimizer = optimizers.Adam(0.00001)
model.compile(optimizer=optimizer,
              loss='mse')
early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1,
                              patience=5, verbose=0, mode='auto',
                              baseline=None, restore_best_weights=False)
history = model.fit(train_x,train_y,validation_data = (test_x,test_y),batch_size =128,epochs =10,callbacks = [early_stopping])

plt.figure()
epochs=range(len(history.history['loss']))
plt.plot(epochs,history.history['loss'],'b',label='Training loss')
plt.plot(epochs,history.history['val_loss'],'r',label='Validation val_loss')
plt.title('Traing and Validation loss')
plt.legend()
plt.show()

"""
inp = test_x[-1000:].reshape((-1,784))
code_x = encoder(inp).numpy()
dt = []
for i in range(10):
    dt.append(code_x[y1[-1000:]==i])
color =['c', 'b', 'g', 'r', 'orange', 'y', 'k', 'silver','pink','purple']
plt.figure(figsize=(12,12))
for i in range(10):
    plot = dt[i]
    plt.scatter(plot[:,0],plot[:,1],c=color[i],label = i)
    plt.legend(loc = 'best')
plt.show()
"""

encoded_imgs = encoder.predict(test_x)
decoded_imgs = decoder.predict(encoded_imgs)

n = 6  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_x[i].reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1+ n)
    plt.imshow(encoded_imgs[i].reshape(10, 10))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1+ n+ n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


"""
intermed_tsne1=tsne.fit_transform(decoded_imgs)
for i in range(10):
    dt.append(intermed_tsne1[y1[-10000:]==i])
color =['c', 'b', 'g', 'r', 'orange', 'y', 'k', 'silver','pink','purple']
plt.figure(figsize=(12,12))
for i in range(10):
    plot = dt[i]
    plt.scatter(plot[:,0],plot[:,1],c=color[i],label = i)
    plt.legend(loc = 'best')
plt.show()
"""