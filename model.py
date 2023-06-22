# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:21:17 2023

@author: Daiiiiiiii
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
from keras.layers import Dense
import time


#Loading mat file containing data
Map = '5'
loadpath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\mapdata' + Map + '.mat'
mappath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\map' + Map + '.png'
mat = scipy.io.loadmat(loadpath)
epochsnum = 40
batch_sizenum = 4


Ranges = mat["Ranges"]
Ranges[Ranges==inf]=10^4
Ydata = mat["Ydata"]
Amcldata = mat["Amcldata"]
Mapinfo = mat["Mapinfo"]
maplen = len(Mapinfo)
Ydata = np.delete(Ydata, -1, axis=1)    # for no theta
Amcldata = np.delete(Amcldata, -1, axis=1)    # for no theta
#Amcldata = np.roll(Amcldata,1,axis=1)
len_ranges = len(np.transpose(Ranges))
val_per = 0.2
# n = int(0.2*len(Ranges))   # number of validation data
n = 100

Ydata_tot = Ydata
AMCL = Amcldata
Ranges_tot = Ranges

#Augmenting data
angle = 60
for i in np.arange(0,361,angle):
    Ranges_roll = np.roll(Ranges, i, axis=1)
    Ranges_tot = np.row_stack((Ranges_tot,Ranges_roll))
    Ydata = np.row_stack((Ydata,Ydata_tot))
    Amcldata = np.row_stack((Amcldata,AMCL))
    
Ranges = Ranges_tot
# Ranges = np.row_stack((Ranges,Ranges))
# Ydata = np.row_stack((Ydata,Ydata))
# Amcldata = np.row_stack((Amcldata,Amcldata))

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


# Split continuously
df_data = df[list(range(1,len_ranges+3))]
df_pose = df[[len_ranges+2, len_ranges+3]]

df_data_tot = pd.DataFrame(scaler_features.fit_transform(df_data), columns = Ranges_label+amcl_label)
df_ranges_tot = df_data_tot[Ranges_label]
df_ranges_val = df_ranges_tot[:n]
df_ranges = df_ranges_tot[n:]
amcl_df = df_data_tot[amcl_label]
amcl_df_val = amcl_df[:n]

df_pose_tot = pd.DataFrame(scaler_labels.fit_transform(df_pose), columns = pos_label)
df_pose = df_pose_tot[n:]
df_pose_val = df_pose_tot[:n]


#The model has 360 inputs for range features:
input_layer = Input(shape=len_ranges,name ='input_layer')

#Hidden layer definition
hidden =  Dense(1028,name='Hidden1', kernel_initializer="random_normal", activation = "relu")(input_layer)
hidden =  Dense(64,name='Hidden2', kernel_initializer="random_normal", activation = "relu")(hidden)

output_layer =  Dense(2, activation= 'linear', name='output_layer')(hidden)

model=keras.Model(inputs=[input_layer], outputs=[output_layer]) 
model.summary()

# define the scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Compiling model
optimizer = tf.keras.optimizers.Nadam(learning_rate = 5E-5)
model.compile(optimizer,loss = 'mean_squared_error'  , metrics=['mae', 'mse'])


training_start_time = time.time()
history = model.fit([df_ranges], [df_pose], 
                     epochs = epochsnum, batch_size=batch_sizenum,
                     verbose=1, validation_split=0.3 ,callbacks=[lr_scheduler])
training_finish_time = time.time()

prediction = model.predict([df_ranges_val])
df_pred = pd.DataFrame(data=prediction, columns = pos_label)
pred_df = pd.DataFrame(scaler_labels.inverse_transform(df_pred), columns = pos_label)
df_pose_val = pd.DataFrame(scaler_labels.inverse_transform(df_pose_val), columns = pos_label)
amcl_df_val = pd.DataFrame(scaler_labels.inverse_transform(amcl_df_val), columns = ['y','x'])
error = abs(pred_df - df_pose_val)
amcl_error = abs(amcl_df_val - df_pose_val)
axis = list(range(0,len(prediction)))
distance = np.sqrt(error["x"]**2+error["y"]**2)

#saving the files and model:
from datetime import datetime
now = datetime.now()
time = now.strftime("%d%m%Y_%H_%M_%S")
dirpath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\map' + Map + 'save_' + time
os.mkdir(dirpath)

savepath = dirpath + '\save_var.mat'
mdict = {
            "pred_df": pred_df.to_numpy(),
            "df_pose_val": df_pose_val.to_numpy(),
            "amcl_df_val": amcl_df_val.to_numpy(),
            "error": error.to_numpy(),
            "amcl_error": amcl_error.to_numpy(),
            "axis": axis,
            "distance": distance.to_numpy()}
scipy.io.savemat(savepath, mdict)


# Use model to predict validation data 
print('Training finished, took {:.2f}s'.format(training_finish_time - training_start_time))
print("Prediction mean error: " + str(error.mean()))
print("AMCL mean error: " + str(amcl_error.mean()))


# Create subplots for each for predict, actual, mean models
figx, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
ax[0].plot(axis, pred_df['x'], label='prediction')
ax[0].plot(axis, df_pose_val['x'], label='actual')
ax[0].plot(axis, amcl_df_val['x'], label='amcl')
ax[0].set_xlabel("x [meters]")
ax[0].set_ylabel("distance")
ax[0].legend()

ax[1].plot(axis, error['x'], label='prediction')
ax[1].plot(axis, amcl_error['y'], label='amcl')
ax[1].set_xlabel("x [meters]")
ax[1].set_ylabel("error")
ax[1].legend()
figx.savefig(dirpath + r"\x_pred.png")

figy, ay = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
ay[0].plot(axis, pred_df['y'], label='prediction')
ay[0].plot(axis, df_pose_val['y'], label='actual')
ay[0].plot(axis, amcl_df_val['y'], label='amcl')
ay[0].set_xlabel("y [meters]")
ay[0].set_ylabel("distance")
ay[0].legend()

ay[1].plot(axis, error['y'], label='prediction')
ay[1].plot(axis, amcl_error['y'], label='amcl')
ay[1].set_xlabel("y [meters]")
ay[1].set_ylabel("error")
ay[1].legend()
figy.savefig(dirpath + r"\y_pred.png")

figloss, aloss = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
aloss[0].plot(history.history['loss'])
aloss[0].plot(history.history['val_loss'])
aloss[0].title.set_label('x loss')
aloss[0].set_ylabel('loss')
aloss[0].set_xlabel('epoch')
aloss[0].legend(['train', 'val'], loc='upper left')

aloss[1].plot(history.history['loss'])
aloss[1].plot(history.history['val_loss'])
aloss[1].title.set_label('y loss')
aloss[1].set_ylabel('loss')
aloss[1].set_xlabel('epoch')
aloss[1].legend(['train', 'val'], loc='upper left')
figloss.savefig(dirpath + r"\train_loss.png")

figscat, ascat = plt.subplots()
a1 = ascat.scatter(pred_df['x'],pred_df['y'], c=distance, cmap = "gist_rainbow_r", s = 20)
ascat.plot(df_pose_val['x'],df_pose_val['y'],label="True positions", color ="black")
ascat.set_ylabel("y [meters]")
ascat.set_xlabel("x [meters]")
ascat.title.set_text("Mapping predicted values x and y")
plt.colorbar(a1, label = 'error [meters]')  
ascat.legend(loc='lower right')
figscat.savefig(dirpath + r"\scatterplot.png")      


# Show the plot
plt.show()
