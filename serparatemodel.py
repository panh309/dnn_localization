# -*- coding: utf-8 -*-
"""
Created on Thu May 25 13:04:51 2023

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
from sklearn.model_selection import train_test_split

#Loading mat file containing data
Map = '5'
loadpath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\mapdata' + Map + '.mat'
mat = scipy.io.loadmat(loadpath)
epochsnum = 100
batch_sizenum = 32


Ranges = mat["Ranges"]
Ranges[Ranges==inf]=10^4
Ydata = mat["Ydata"]
Amcldata = mat["Amcldata"]
Mapinfo = mat["Mapinfo"]
maplen = len(Mapinfo)
Ydata = np.delete(Ydata, -1, axis=1)    # for no theta
Amcldata = np.delete(Amcldata, -1, axis=1)    # for no theta
len_ranges = len(np.transpose(Ranges))
val_per = 0.2
n = int(0.2*len(Ranges))   # number of validation data
Mapinfo = Mapinfo*len_ranges
df_map = pd.DataFrame(data=Mapinfo)

Ydata_tot = Ydata
AMCL = Amcldata
Ranges_tot = Ranges

angle = 45
for i in np.arange(0,361,angle):
    Ranges_roll = np.roll(Ranges, i, axis=1)
    Ranges_tot = np.row_stack((Ranges_tot,Ranges_roll))
    Ydata = np.row_stack((Ydata,Ydata_tot))
    Amcldata = np.row_stack((Amcldata,AMCL))
    
Ranges = Ranges_tot

Ranges = np.row_stack((Ranges,Ranges))
Ydata = np.row_stack((Ydata,Ydata))
Amcldata = np.row_stack((Amcldata,Amcldata))
# Ranges1 = np.roll(Ranges, 90, axis=1)
# Ranges2 = np.roll(Ranges, 180, axis=1)
# Ranges3 = np.roll(Ranges, 270, axis=1)
# Ranges = np.row_stack((Ranges,Ranges1, Ranges2, Ranges3))
# Ydata = np.row_stack((Ydata,Ydata,Ydata,Ydata))
# Amcldata = np.row_stack((Amcldata,Amcldata,Amcldata,Amcldata))
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


# df_data_tot = pd.DataFrame(scaler_features.fit_transform(df_data), columns = Ranges_label+amcl_label)

# df_pose_tot = pd.DataFrame(scaler_labels.fit_transform(df_pose), columns = pos_label)

# df_data, df_data_val, df_pose, df_pose_val = train_test_split(df_data_tot, df_pose_tot, test_size=val_per, random_state=42)

# amcl_df_val = df_data_val[amcl_label]
# df_ranges = df_data[Ranges_label]
# df_ranges_val = df_data_val[Ranges_label]

# Split continuously

df_data_tot = pd.DataFrame(scaler_features.fit_transform(df_data), columns = Ranges_label+amcl_label)
df_ranges_tot = df_data_tot[Ranges_label]
df_ranges_val = df_ranges_tot[:n]
df_ranges = df_ranges_tot[n:]
amcl_df = df_data_tot[amcl_label]
amcl_df_val = amcl_df[:n]

df_pose_tot = pd.DataFrame(scaler_labels.fit_transform(df_pose), columns = pos_label)
df_pose = df_pose_tot[n:]
df_pose_val = df_pose_tot[:n]

from datetime import datetime
now = datetime.now()
time = now.strftime("%d%m%Y_%H_%M_%S")
dirpath = r'C:\Users\panh3\Desktop\Rosbag3\map' + Map + '\map' + Map + 'save_' + time
os.mkdir(dirpath)

def trainmodel(var):
    keras.backend.clear_session()
    input_layer = Input(shape=len_ranges,name ='input_layer')
    
    #Hidden layer definition
    hidden =  Dense(256,name='Hidden1', kernel_initializer="random_normal", activation = "tanh")(input_layer)
    hidden =  Dense(64,name='Hidden2', kernel_initializer="random_normal", activation = "tanh")(hidden)
    
    output_layer =  Dense(1, activation= 'linear', name='output_layer')(hidden)
    
    model=keras.Model(inputs=[input_layer], outputs=[output_layer]) 
    
    # define the scheduler
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    
    # Compiling model
    optimizer = tf.keras.optimizers.Adam(learning_rate = 5E-5)
    model.compile(optimizer,loss = 'mean_squared_error'  , metrics=['mae', 'mse'])
    
    history = model.fit([df_ranges], [df_pose[var]], 
                        epochs = epochsnum, batch_size=batch_sizenum,
                        verbose=1, validation_split=0.3 ,callbacks=[lr_scheduler])
    modelpath = dirpath + '\model_' + var + '.h5'
    model.save(modelpath)
    return history
model_x = trainmodel("x")
model_y = trainmodel("y")

prediction_x = model_x.model.predict([df_ranges_val])
prediction_y = model_y.model.predict([df_ranges_val])

prediction = np.column_stack((prediction_x, prediction_y))
df_pred = pd.DataFrame(data=prediction, columns = pos_label)
pred_df = pd.DataFrame(scaler_labels.inverse_transform(df_pred), columns = pos_label)
df_pose_val = pd.DataFrame(scaler_labels.inverse_transform(df_pose_val), columns = pos_label)
amcl_df_val = pd.DataFrame(scaler_labels.inverse_transform(amcl_df_val), columns = pos_label)
error = abs(pred_df - df_pose_val)
amcl_error = abs(amcl_df_val - df_pose_val)
axis = list(range(0,len(prediction)))
distance = np.sqrt(error["x"]**2+error["y"]**2)


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
print("Prediction mean error: " + str(error.mean()))
print("AMCL mean error: " + str(amcl_error.mean()))


# Create subplots for each for predict, actual, mean models
figx, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
ax[0].plot(axis, pred_df['x'], label='prediction')
ax[0].plot(axis, df_pose_val['x'], label='actual')
#ax[0].plot(axis, amcl_df_val['x'], label='amcl')
ax[0].set_ylabel("x")
ax[0].set_xlabel("distance")
ax[0].legend()

ax[1].plot(axis, error['x'], label='prediction')
ax[1].plot(axis, amcl_error['y'], label='amcl')
ax[1].set_ylabel("x")
ax[1].set_xlabel("error")
ax[1].legend()
figx.savefig(dirpath + r"\x_pred.png")

figy, ay = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
ay[0].plot(axis, pred_df['y'], label='prediction')
ay[0].plot(axis, df_pose_val['y'], label='actual')
#ay[0].plot(axis, amcl_df_val['y'], label='amcl')
ay[0].set_ylabel("y")
ay[0].set_xlabel("distance")
ay[0].legend()

ay[1].plot(axis, error['y'], label='prediction')
ay[1].plot(axis, amcl_error['y'], label='amcl')
ay[1].set_ylabel("y")
ay[1].set_xlabel("error")
ay[1].legend()
figy.savefig(dirpath + r"\y_pred.png")

figloss, aloss = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
aloss[0].plot(model_x.history['loss'])
aloss[0].plot(model_x.history['val_loss'])
aloss[0].title.set_label('x loss')
aloss[0].set_ylabel('loss')
aloss[0].set_xlabel('epoch')
aloss[0].legend(['train', 'val'], loc='upper left')

aloss[1].plot(model_y.history['loss'])
aloss[1].plot(model_y.history['val_loss'])
aloss[1].title.set_label('y loss')
aloss[1].set_ylabel('loss')
aloss[1].set_xlabel('epoch')
aloss[1].legend(['train', 'val'], loc='upper left')
figloss.savefig(dirpath + r"\train_loss.png")

figscatx, ascatx = plt.subplots()
bx = ascatx.scatter(pred_df['x'],df_pose_val['y'], c=error["x"], cmap = "gist_rainbow_r")
ascatx.scatter(df_pose_val['x'],df_pose_val['y'], marker="x")
ascatx.set_ylabel("y")
ascatx.set_xlabel("x")
ascatx.title.set_text("Mapping predicted values x with true values of y")
plt.colorbar(bx, label = 'error')  
figscatx.savefig(dirpath + r"\scatterpredx.png")     

figscaty, ascaty = plt.subplots()
by = ascaty.scatter(df_pose_val['x'],pred_df['y'], c=error["y"], cmap = "gist_rainbow_r")
ascaty.scatter(df_pose_val['x'],df_pose_val['y'], marker="x")
ascaty.set_ylabel("y")
ascaty.set_xlabel("x")
ascaty.title.set_text("Mapping predicted values y with true values of x")
plt.colorbar(by, label = 'error')  
figscaty.savefig(dirpath + r"\scatterpredy.png")     

figscat, ascat = plt.subplots()
a1 = ascat.scatter(pred_df['x'],pred_df['y'], c=distance, cmap = "gist_rainbow_r")
ascat.scatter(df_pose_val['x'],df_pose_val['y'], marker="x")
ascat.set_ylabel("y")
ascat.set_xlabel("x")
ascat.title.set_text("Mapping predicted values x and y")
plt.colorbar(a1, label = 'error')  
figscat.savefig(dirpath + r"\scatterplot.png")     

# Show the plot
plt.show()
