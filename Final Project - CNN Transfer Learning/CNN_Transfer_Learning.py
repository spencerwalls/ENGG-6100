# Spencer Walls, 0734584
# This program is for developing transfer learning models
# to be trained and tested on the UC Merced Land-Use dataset.

import numpy as np
from keras import utils
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# image dimensions 
img_width, img_height = 256, 256

# path to the weights of the most optimally performing model,
# these are saved throughout the training phase 
top_model_weights_path = 'bottleneck_fc_model_mv1.h5'

train_data_dir = 'data1/train1' # directories of training and testing data
validation_data_dir = 'data1/validation1'
nb_train_samples = 1680 # number of training samples
nb_validation_samples = 420 # number of testing samples 
epochs = 10
batch_size = 16

# This function extracts features from the training and testing images
# using a state-of-the-art CNN architecture. the VGG19, Xception, and 
# MobileNet architectures were all individually implemented. 
def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # this command instantiates the state-of-the-art CNN architecture that will 
    # be used for transfer learning
    model = applications.mobilenet.MobileNet(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    
    np.save('bottleneck_features_train_mv1.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    
    np.save('bottleneck_features_validation_mv1.npy',
            bottleneck_features_validation)

# call the above function to execute the feature extraction
save_bottleneck_features()

### training phase begins ###
# the code found below pertains to the fully-connected section at the end of
# the CNN model. after the features have been extracted by one of the 
# state-of-the-art architectures, the fully-connected section, which is 
# essentially a classic Multilayer Perceptron, may be trained using as input
# these features that have been extracted. 
train_data = np.load('bottleneck_features_train_mv1.npy') # load training inputs 

train_labels = np.zeros((1680,21))
j = 0
i = 0
for j in range(0, 21):
    train_labels[i:i+80, j] = 1
    i = i+80

validation_data = np.load('bottleneck_features_validation_mv1.npy') # load testing inputs

validation_labels = np.zeros((420,21))
j = 0
i = 0
for j in range(0, 21):
    validation_labels[i:i+20, j] = 1
    i = i+20

# one hidden layer is used, and 21 output neurons pertaining to the 21 classes
model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(21, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# these callbacks are responsible for saving the weights of the model as it improves
# as well as terminating the execution of the training phase if the model goes through
# 6 consecutive training epochs without improving. 
call_backs = [ModelCheckpoint(top_model_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
              EarlyStopping(monitor='val_acc', patience=6, verbose=0)]

# training the model 
model.fit(train_data, train_labels,
          epochs=100,
          batch_size=16,
          validation_data=(validation_data, validation_labels),
          callbacks=call_backs)
### training phase ends ###

### testing phase begins ###
model.load_weights(top_model_weights_path)

Y_test = np.argmax(validation_labels, axis=1) # convert from one-hot 
                   
Y_pred = model.predict_classes(validation_data) # predict classes of unseen images 

print(classification_report(Y_test, Y_pred)) # view classification report

print("accuracy = ", accuracy_score(Y_test, Y_pred)) # view model accuracy 

confusion_matrix1 = confusion_matrix(Y_test, Y_pred) # create confusion matrix
confusion_matrix_weighted = (confusion_matrix1 / 20) * 100 # convert matrix to % 
