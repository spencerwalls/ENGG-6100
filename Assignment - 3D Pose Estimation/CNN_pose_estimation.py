# This program develops a CNN for regression.

# Import libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import cv2
import os
import glob

# Import images (X - independent variable)
img_dir = "all_images"  
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

# Import poses of objects (Y - dependent variables)
dataset = pd.read_csv("object_poses.csv").values


cheeseY = dataset[:420, 1:]
crayolaY = dataset[420:840, 1:]
expoY = dataset[840:1272, 1:]
felineY = dataset[1272:1692, 1:]
genuineY = dataset[1692:2124, 1:]

cheeseX = data[:420]
crayolaX = data[420:840]
expoX = data[840:1272]
felineX = data[1272:1692]
genuineX = data[1692:2124]

# Split the data into training and testing sets 
(cheeseYtrain, cheeseYtest, cheeseXtrain, cheeseXtest) = train_test_split(cheeseY, cheeseX, test_size=0.25)
(crayolaYtrain, crayolaYtest, crayolaXtrain, crayolaXtest) = train_test_split(crayolaY, crayolaX, test_size=0.25)
(expoYtrain, expoYtest, expoXtrain, expoXtest) = train_test_split(expoY, expoX, test_size=0.25)
(felineYtrain, felineYtest, felineXtrain, felineXtest) = train_test_split(felineY, felineX, test_size=0.25)
(genuineYtrain, genuineYtest, genuineXtrain, genuineXtest) = train_test_split(genuineY, genuineX, test_size=0.25)

#append matrices for x and y train
total_Xtrain = cheeseXtrain + crayolaXtrain + expoXtrain + felineXtrain + genuineXtrain

total_Ytrain = np.zeros((1593,12))
total_Ytrain[0:315,:] = cheeseYtrain
total_Ytrain[315:630,:] = crayolaYtrain
total_Ytrain[630:954,:] = expoYtrain
total_Ytrain[954:1269,:] = felineYtrain
total_Ytrain[1269:1593] = genuineYtrain

(final_Ytrain, NULL1, final_Xtrain, NULL2) = train_test_split(total_Ytrain, total_Xtrain, test_size=0)


## TEST REGION #
#split = train_test_split(dataset[840:1272,1:], data[840:1272], test_size=0.25)
#(Y_train, Y_test, X_train, X_test) = split

final_Xtrain_scaled = (np.array(final_Xtrain))/255

# scale all Xtest data for all 5 objects  
cheeseXtest_scaled = (np.array(cheeseXtest))/255
crayolaXtest_scaled = (np.array(crayolaXtest))/255
expoXtest_scaled = (np.array(expoXtest))/255
felineXtest_scaled = (np.array(felineXtest))/255
genuineXtest_scaled = (np.array(genuineXtest))/255

#X_test_scaled = (np.array(X_test))/255

# TEST REGION END #

# Initialize CNN
regressor = Sequential()

# Step 1 - Convolution
regressor.add(Convolution2D(filters = 32, kernel_size =  3, strides = 3, input_shape = (480, 640, 3), activation = 'relu'))

# Step 2 - Pooling
regressor.add(MaxPooling2D(pool_size = (2,2)))

# Add second Conv layer
regressor.add(Convolution2D(32, 3, 3, activation = 'relu'))
regressor.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
regressor.add(Flatten())

# Step 4 - Full connection
regressor.add(Dense(units = 128, activation = 'relu'))
regressor.add(Dense(units = 12, activation = 'linear'))

# Step 5 - Compile
regressor.compile(optimizer = 'adam', loss = 'MSE', metrics = ['MAE'])

# Step 6 - Fit CNN to the images
regressor.fit(final_Xtrain_scaled, final_Ytrain, batch_size=10, epochs=10,verbose=1)

# Step 7 - Predict the pose of objects in unseen images 
y_pred = regressor.predict(genuineXtest_scaled, batch_size=None, verbose=2, steps=None)

# Step 8 - Evaluate the model 
# This step is given in the L2dist_compute.py file 





