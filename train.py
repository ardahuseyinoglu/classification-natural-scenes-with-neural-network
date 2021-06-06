### importing libraries

import numpy as np
import pandas as pd
import argparse

# sklearn 
from sklearn.preprocessing import LabelEncoder

# keras
from keras.layers import Input
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as prep_input_resnet50

# read images
import os
from tqdm import tqdm 
import pickle

# data preparation and neural network model
import prepare_data
from nn_model import NeuralNetwork, Layer


#-------------------------------------------------------------------

### get command line arguments

ap = argparse.ArgumentParser()

ap.add_argument("-train_data_path", "--train_data_path", required=True, help="provide the training dataset path")
ap.add_argument("-valid_data_path", "--valid_data_path", required=True, help="provide the validation dataset path")
ap.add_argument("-img_feature", "--img_feature", required=True, help="provide the feature extraction method for images --> resized_gray or deep_resnet50")

args = vars(ap.parse_args())
train_data_path = args['train_data_path']
valid_data_path = args['valid_data_path']
img_feature = args['img_feature']

#-------------------------------------------------------------------

### loading training and validation dataset

seg_train_path = train_data_path.split('\\')[1:]
seg_train_path = os.path.join(*seg_train_path)

seg_dev_path = valid_data_path.split('\\')[1:]
seg_dev_path = os.path.join(*seg_dev_path)

print("\nTrain images are being read...")
X_train, Y_train, number_of_samples_train = prepare_data.load_data(path=seg_train_path)
print("\nValidation images are being read...")
X_valid, Y_valid, number_of_samples_valid = prepare_data.load_data(path=seg_dev_path)

#------------------------------------------------------------------

# feature extraction from images and encoding target values

if img_feature == 'resized_gray':
    print("\nTraining image features are being prepared...")
    prepare_data.convert_to_resized_gray(X_train, 30, 'resized_gray_train.npz')
    print("\nValidation image features are being prepared...")
    prepare_data.convert_to_resized_gray(X_valid, 30, 'resized_gray_valid.npz')

    
elif img_feature == 'deep_resnet50':
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')
    
    print("\nTraining image features are being prepared...")
    prepare_data.convert_deep_features(X_train, model, prep_input_resnet50, 'resnet50_train.npz')
    print("\nValidation image features are being prepared...")
    prepare_data.convert_deep_features(X_valid, model, prep_input_resnet50, 'resnet50_valid.npz')


le = LabelEncoder()
le.fit(Y_train)
Y_train_encoded = le.transform(Y_train)
Y_train_encoded = np.array(pd.get_dummies(Y_train_encoded))

le = LabelEncoder()
le.fit(Y_valid)
Y_valid_encoded = le.transform(Y_valid)
Y_valid_encoded = np.array(pd.get_dummies(Y_valid_encoded))

np.savez_compressed('Y_train_encoded.npz', Y_train_encoded)
np.savez_compressed('Y_valid_encoded.npz', Y_valid_encoded)


#----------------------------------------------------------------------

# getting data ready for training: loading and shuffling

if img_feature == 'deep_resnet50':
    X_train_ = np.load('resnet50_train.npz')['arr_0']
    X_valid_ = np.load('resnet50_valid.npz')['arr_0']
    
elif img_feature == 'resized_gray':
    X_train_ = np.load('resized_gray_train.npz')['arr_0']
    X_valid_ = np.load('resized_gray_valid.npz')['arr_0']


Y_train_encoded = np.load('Y_train_encoded.npz')['arr_0']
Y_valid_encoded = np.load('Y_valid_encoded.npz')['arr_0']


# shuffling
random_index = np.arange(len(X_train_))
np.random.shuffle(random_index)
X_train_ = X_train_[random_index]
Y_train_encoded = Y_train_encoded[random_index]

random_index = np.arange(len(X_valid_))
np.random.shuffle(random_index)
X_valid_ = X_valid_[random_index]
Y_valid_encoded = Y_valid_encoded[random_index]

#-----------------------------------------------------------------------

# TRAINING

# set nn input size
nn_input_size = 2048
if img_feature == 'resized_gray':
    nn_input_size = 900

# data
x_train = X_train_[:]
y_train = Y_train_encoded[:]
x_valid = X_valid_[:]
y_valid = Y_valid_encoded[:]

# build model
model = NeuralNetwork('cross_entropy')
model.add(Layer(nn_input_size,512, 'relu'))
model.add(Layer(512,128, 'relu'))
model.add(Layer(128,64, 'relu'))
model.add(Layer(64,6, 'softmax'))

# hyperparameters
num_epochs = 50
batch_size = 32
lr = 0.1
lr_decay_params = {'drop_amount':0.75, 
                   'num_epochs_to_drop': 10}

# training
training_accs, validation_accs, training_losses, validation_losses = model.train(x_train, 
                                                                               y_train, 
                                                                               x_valid, 
                                                                               y_valid, 
                                                                               epochs = num_epochs, 
                                                                               batch_size = batch_size, 
                                                                               learning_rate = lr, 
                                                                               lr_decay_params = lr_decay_params)


# saving the model
pickle.dump(model, open('new_model.sav', 'wb'))

