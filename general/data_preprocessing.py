### Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#keras
from keras.preprocessing import image
from keras.layers import merge, Input

# read images
import cv2 
import os
from tqdm import tqdm 


### Loading Dataset

def load_class_data(class_dir_path, class_dir_name, X, Y):
    names_of_imgs = os.listdir(class_dir_path)
    names_of_imgs = sorted(names_of_imgs, key=lambda x: int(os.path.splitext(x)[0]))
    counter = 0
    for img_name in tqdm(names_of_imgs, ascii=True, desc=class_dir_name):
        img_path = os.path.join(class_dir_path,img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(150, 150))
        X.append(np.array(img))
        Y.append(class_dir_name)
        counter += 1
    return counter


def load_data(path):
    X = []
    Y = []
    counts_class_samples = []
    for class_dir_name in os.listdir(path):
        class_dir_path = os.path.join(path,class_dir_name)
        number_of_sample = load_class_data(class_dir_path, class_dir_name, X, Y)
        counts_class_samples.append(number_of_sample)
    return X, Y, counts_class_samples



seg_train_path = os.path.join('data','seg_train')
seg_dev_path = os.path.join('data','seg_dev')
seg_test_path = os.path.join('data','seg_test')

classes = os.listdir(seg_train_path)

X_train, Y_train, number_of_samples_train = load_data(path=seg_train_path)
X_valid, Y_valid, number_of_samples_valid = load_data(path=seg_dev_path)
X_test = []

names_of_imgs = os.listdir(seg_test_path)
names_of_imgs = sorted(names_of_imgs, key=lambda x: int(os.path.splitext(x)[0]))

for img_name in tqdm(names_of_imgs, ascii=True):
        img_path = os.path.join(seg_test_path,img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(150, 150))
        X_test.append(np.array(img))


### Sample Images
fig,ax=plt.subplots(2,6)
fig.set_size_inches(15,15)

ax[0,0].imshow(X_train[37])
ax[0,0].set_title("buildings")
ax[0,0].grid(None)

ax[0,1].imshow(X_train[2601])
ax[0,1].set_title("forest")
ax[0,1].grid(None)

ax[0,2].imshow(X_train[5000])
ax[0,2].set_title("glacier")
ax[0,2].grid(None)

ax[0,3].imshow(X_train[8259])
ax[0,3].set_title("mountain")
ax[0,3].grid(None)

ax[0,4].imshow(X_train[10357])
ax[0,4].set_title("sea")
ax[0,4].grid(None)

ax[0,5].imshow(X_train[12567])
ax[0,5].set_title("street")
ax[0,5].grid(None)

plt.tight_layout()


### Visualize Distribution

sns.barplot(x=classes, y=number_of_samples_train, palette="pastel").set(title='Train Set Distribution')


### Preparing data

#### Features

###### Resized & Gray Scale  (outshape: 900)

def convert_to_resized_gray(data, resize_size, compressed_file_name):
    converted_data = np.empty([len(data), resize_size*resize_size])
    
    for i in tqdm(range(len(data))):
        # resize
        converted_img = cv2.resize(data[i], dsize=(resize_size, resize_size), interpolation=cv2.INTER_CUBIC)
        # convert to grayscale
        converted_img = cv2.cvtColor(converted_img, cv2.COLOR_RGB2GRAY)
        # flatten 
        converted_img = converted_img.flatten()
        # normalize
        converted_img = converted_img / 255
        converted_data[i] = converted_img
        
    # to save feature np.array to a csv.file:
    np.savez_compressed(compressed_file_name, converted_data)

    
convert_to_resized_gray(X_train, 30, 'resized_gray_train.npz')
convert_to_resized_gray(X_valid, 30, 'resized_gray_valid.npz')
convert_to_resized_gray(X_test, 30, 'resized_gray_test.npz')


##### Deep Features (Feature extraction using CNN with Transfer Learning)

def convert_deep_features(data, model, preprocess_input, compressed_file_name):
    converted_data = np.empty([len(data), model.output_shape[1]])
    
    for i in tqdm(range(len(data))):
        #normalized_img = data[i] / 255
        #img_data_expanded = np.expand_dims(normalized_img, axis=0)
        img_data_expanded = np.expand_dims(data[i], axis=0)
        image_processed = preprocess_input(img_data_expanded)
        deep_features = model.predict(image_processed)
        deep_features = deep_features.flatten()
        converted_data[i] = deep_features

    # to save feature np.array to a csv.file:
    np.savez_compressed(compressed_file_name, converted_data)


###### VGG-16 (outshape: 512)

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as prep_input_vgg16

model = VGG16(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')

convert_deep_features(X_train, model, prep_input_vgg16, 'vgg16_train.npz')
convert_deep_features(X_valid, model, prep_input_vgg16, 'vgg16_valid.npz')
convert_deep_features(X_test, model, prep_input_vgg16, 'vgg16_test.npz')


###### VGG-19 (outshape: 512)
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as prep_input_vgg19

model = VGG19(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')

convert_deep_features(X_train, model, prep_input_vgg19, 'vgg19_train.npz')
convert_deep_features(X_valid, model, prep_input_vgg19, 'vgg19_valid.npz')
convert_deep_features(X_test, model, prep_input_vgg19, 'vgg19_test.npz')


###### ResNet-50 (outshape: 2048)
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as prep_input_resnet50

model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')

convert_deep_features(X_train, model, prep_input_resnet50, 'resnet50_train.npz')
convert_deep_features(X_valid, model, prep_input_resnet50, 'resnet50_valid.npz')
convert_deep_features(X_test, model, prep_input_resnet50, 'resnet50_test.npz')


###### InceptionV3 (outshape: 2048)
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as prep_input_inceptionv3

model = InceptionV3(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')

convert_deep_features(X_train, model, prep_input_inceptionv3, 'inceptionv3_train.npz')
convert_deep_features(X_valid, model, prep_input_inceptionv3, 'inceptionv3_valid.npz')
convert_deep_features(X_test, model, prep_input_inceptionv3, 'inceptionv3_test.npz')


# ## Encoding target values
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

