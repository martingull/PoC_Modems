#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:21:49 2017
Original program created by martin.gullaksen using libraries Tensorflow and tflearn
Modified to use libraries Theano and keras and to include post-analysis by laura.astola
"""

from __future__ import division, print_function, absolute_import

import numpy as np
import os

from keras.models import load_model
from keras.optimizers import SGD
from DataProcessing import PreProcessor
from NetworkGenerator import NetworkGenerator
from sklearn.model_selection import train_test_split

# Set to 'TRAIN' or 'TEST'
mode = 'TRAIN'  
#mode = 'TEST'

train_data_folder='./processed_data'
test_data_folder='./processed_data_test'
model_folder='./learned_models'

dataset = PreProcessor() 

#  extracts raw images from video, e.g. by capturing every 24th frame
#  does not have to be run over and over again
#dataset.video_to_images(image_path='./raw_pictures', video_path='./trimmed_videos',\
#                        size_pics=(256, 256), frame_spacing=12) 
#dataset.video_to_images(image_path='./raw_pictures_test', video_path='./Leave_one_out',\
#                        size_pics=(256, 256), frame_spacing=24) 


#  augments the (manually cleaned) raw images to form a richer training set
#  does not have to be run over and over again
#dataset.augment_data(image_path='./raw_pictures', augment_path='./processed_data', \
#                     batch_size=3)
#dataset.augment_data(image_path='./raw_pictures_test', augment_path='./processed_data_test',\
#                     batch_size=3)

#  gives the size of the images to be classified 
size_pics = dataset.size_pics 
   
net_gen = NetworkGenerator()

#  choose one of the different network architectures
model, net_name = net_gen.get_vgg_net(pic_size=size_pics)
#network, net_name = net_gen.get_convnet(pic_size=size_pics)
#network, net_name = net_gen.get_alex_net(pic_size=size_pics)
#network, net_name = net_gen.get_highway(pic_size=size_pics)

#  choose the parameters for the network model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,\
              metrics=['mae','acc'])

#********************************training
if mode == 'TRAIN':
    
#  loads the training data as well as their labels
    X, y = dataset.load_data(train_data_folder)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#   fit a model    
    model.fit(X_train, y_train, batch_size=32, epochs=1)
    scores = model.evaluate(X_test, y_test, batch_size=32,verbose=2)

#  check if destination folder present
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)    
    model.save(model_folder+'/model.h5')


#*****************************testing
elif mode == 'TEST':
    
# load the model fitted with training data
    
    model=load_model(model_folder+'/model.h5')

##*****************************post-analysis

    folders = [folder for folder in os.listdir(test_data_folder)]
    prediction_accuracies=[]
    folder_start_index=0
    col_number=0
    
    for folder in folders:
# loads the testing data as well as their labels
        X, Y = dataset.load_data(test_data_folder)
        number_of_predictions=len(os.listdir(test_data_folder+'/'+folder)) 
        
        Y_pred=model.predict(X[folder_start_index:folder_start_index+number_of_predictions])
        
# to compute binarized accuracy: set the class with highest prob. to one, rest to zero       
        Y_class=np.zeros_like(Y_pred)
        Y_class[np.arange(len(Y_pred)),Y_pred.argmax(1)]=1

        accuracy=(1-sum(abs(Y[folder_start_index:folder_start_index+number_of_predictions,col_number]
                            -Y_class[:,col_number]))/number_of_predictions)*100
        
        print("real statuses = " + str(np.sum(Y[folder_start_index:folder_start_index+number_of_predictions],axis=0)))
        print("predicted statuses = " + str(np.sum(Y_class,axis=0)))
        print('accurately predicted class %s in %.2f percent of the cases' %(folder,accuracy))
        
        prediction_accuracies.append(accuracy)
        folder_start_index+=number_of_predictions
        col_number+=1
