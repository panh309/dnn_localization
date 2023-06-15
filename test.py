# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:29:59 2023

@author: panh3
"""

import os
import scipy.io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from numpy import inf
import pandas as pd
from keras import Input
from keras.layers import Dense, Flatten, Dropout, Concatenate
from PIL import Image
from sklearn.model_selection import train_test_split

#Loading mat file containing data
Map = '3'
loadpath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\mapdata' + Map + '.mat'
mappath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\map' + Map + '.png'
mat = scipy.io.loadmat(loadpath)
im = Image.open(mappath)
epochsnum = 100
batch_sizenum = 32


Ranges = mat["Ranges"]
Ranges[Ranges==inf]=10^4
Ydata = mat["Ydata"]
Amcldata = mat["Amcldata"]
Ydata = np.delete(Ydata, -1, axis=1)    # for no theta
Amcldata = np.delete(Amcldata, -1, axis=1)    # for no theta
len_ranges = len(np.transpose(Ranges))
val_per = 0.2

Ranges1 = np.roll(Ranges, 90, axis=1)
Ranges2 = np.roll(Ranges, 180, axis=1)
Ranges3 = np.roll(Ranges, 270, axis=1)
Ranges = np.row_stack((Ranges,Ranges1, Ranges2, Ranges3))
Ydata = np.row_stack((Ydata,Ydata,Ydata,Ydata))
Amcldata = np.row_stack((Amcldata,Amcldata,Amcldata,Amcldata))

# Creating Ranges dataframe for data
Ranges_label= list(map(str,list(range(1,len_ranges+1))))
amcl_label = ['xamcl', 'yamcl']
pos_label = ['x', 'y']
data_combined = np.column_stack((Ranges,Amcldata, Ydata))
df = pd.DataFrame(data=data_combined)

# Scaling, Dividing training and validation data
from sklearn import preprocessing
scaler_features = preprocessing.StandardScaler()
scaler_labels = preprocessing.StandardScaler()

# Split randomly
df_data = df[list(range(1,len_ranges+3))]
df_pose = df[[len_ranges+2, len_ranges+3]]


df_data_tot = pd.DataFrame(scaler_features.fit_transform(df_data), columns = Ranges_label+amcl_label)

df_pose_tot = pd.DataFrame(scaler_labels.fit_transform(df_pose), columns = pos_label)

df_data, df_data_val, df_pose, df_pose_val = train_test_split(df_data_tot, df_pose_tot, test_size=val_per, random_state=42)

amcl_df_val = df_data_val[amcl_label]
df_ranges = df_data[Ranges_label]
df_ranges_val = df_data_val[Ranges_label]

input_layer = Input(shape=len_ranges,name ='input_layer')

#Hidden layer definition
hidden =  Dense(256,name='Hidden1', kernel_initializer="random_normal", activation = "linear")(input_layer)
hidden =  Dense(64,name='Hidden2', kernel_initializer="random_normal", activation = "linear")(hidden)

output_layer =  Dense(1, activation= 'linear', name='output_layer')(hidden)

model_x=keras.Model(inputs=[input_layer], outputs=[output_layer]) 

# define the scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Compiling model
optimizer = tf.keras.optimizers.Nadam(learning_rate = 5E-5)
model_x.compile(optimizer,loss = 'mean_squared_error'  , metrics=['mae', 'mse'])

history = model_x.fit([df_ranges], [df_pose['y']], 
                    epochs = epochsnum, batch_size=batch_sizenum,
                    verbose=1, validation_split=0.3 ,callbacks=[lr_scheduler])


prediction_y = history.model.predict([df_ranges_val])
error = abs(prediction_y-df_pose_val['y'].to_numpy().reshape((len(prediction_y),1)))
print(error.mean())