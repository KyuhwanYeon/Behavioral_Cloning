#%% Load libraries

from urllib.request import urlretrieve
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import sklearn
import matplotlib.pyplot as plt


#%% Load data
samples = [] #simple array to append all the entries present in the .csv file
with open('./data/data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)

# split dataset -> train:0.85, validation: 0.15

train_samples, validation_samples = train_test_split(samples,test_size=0.15)
#%% Data generation

def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) #shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # in batch_sample, there are 3 samples [center, left, right]
                for i in range(0,3): 
                    name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                    #since cv2 read image as BGR, we need to convert it as RGB
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                    #bat_sample[3] is measured steering angle
                    center_angle = float(batch_sample[3]) 
                    images.append(center_image)
                    
                    # in order to use left and right image as training data, add 0.2 steering angle to left and -0.2 to right image
                    if(i==0):
                        angles.append(center_angle)
                    elif(i==1):
                        angles.append(center_angle+0.2)
                    elif(i==2):
                        angles.append(center_angle-0.2)
                
                    # add horizontally flipped image  
                    images.append(cv2.flip(center_image,1))
                    if(i==0):
                        angles.append(center_angle*-1)
                    elif(i==1):
                        angles.append((center_angle+0.2)*-1)
                    elif(i==2):
                        angles.append((center_angle-0.2)*-1)

            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield (X_train, y_train) 
            

# compile and train the model using the generator function



train_generator = generator(train_samples, batch_size=1)
validation_generator = generator(validation_samples, batch_size=1)


#%% Model design
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D


def NVIDIA_PILOT_NET():
    model = Sequential()
    # Add lambda layer to normalize the pixeles 0~255 to -0.5~0.6
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop images to exclude sky and hood 
    model.add(Cropping2D(cropping=((70,25),(0,0))))           
    # 1 Convolution layer  (filter num: 24, filter size: 5x5, stride: 2x2)
    model.add(Conv2D(24, 5, 2, input_shape = (66, 200, 3), activation = 'relu'))
    # 2 Convolution layer  (filter num: 36, filter size: 5x5, stride: 2x2)
    model.add(Conv2D(36, 5, 2, activation = 'relu'))
    # 3 Convolution layer  (filter num: 48, filter size: 5x5, stride: 2x2)
    model.add(Conv2D(48, 5, 2, activation = 'relu'))
    # 4 Convolution layer  (filter num: 64, filter size: 3x3, stride: 1x1)
    model.add(Conv2D(64, 3, activation = 'relu'))
    # 5 Convolution layer  (filter num: 64, filter size: 3x3, stride: 1x1)
    model.add(Conv2D(64, 3, activation = 'relu'))
    #flatten image from 2D to side by side
    model.add(Flatten())
    # 6 Fully connected layer
    model.add(Dense(100))
    model.add(Activation('elu'))
    # Dropout for avoiding overfitting
    model.add(Dropout(0.25))
    # 7 Fully connected layer
    model.add(Dense(50))
    model.add(Activation('elu'))
    # 8 Fully connected layer
    model.add(Dense(10))
    model.add(Activation('elu'))
    # 9 Fully connected layer
    # Output is only one value which is measured steer angle
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')
    return model

#%% Main
if __name__ == '__main__':
    model = NVIDIA_PILOT_NET()
    history_object = model.fit_generator( train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator,   validation_steps=len(validation_samples), epochs=5, verbose=1)

    # save model
    model.save('model.h5')
    
    # print summary
    model.summary()
    
    # plot
    ### print the keys contained in the history object
    print(history_object.history.keys())
    #%%
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.grid('on')
    plt.savefig('img/loss.png')
    plt.show()
    



