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
import cv2

# data preparation and neural network model
import prepare_data
import nn_model

#-------------------------------------------------------------------

### get command line arguments

ap = argparse.ArgumentParser()

ap.add_argument("-test_data_path", "--test_data_path", required=True, help="provide the test dataset path")
ap.add_argument("-model_file", "--model_file", required=True, help="provide the saved neural network model object (.sav)")
ap.add_argument("-target_values", "--target_values",  help="provide the file(.csv) of ground truth values of test samples")


args = vars(ap.parse_args())
test_data_path = args['test_data_path']
model_file = args['model_file']
target_values = args['target_values']

#-------------------------------------------------------------------

### loading training dataset, model, target values(if provided)

seg_test_path = test_data_path.split('\\')[1:]
seg_test_path = os.path.join(*seg_test_path)


X_test = []

names_of_imgs = os.listdir(seg_test_path)
names_of_imgs = sorted(names_of_imgs, key=lambda x: int(os.path.splitext(x)[0]))

for img_name in tqdm(names_of_imgs, ascii=True):
        img_path = os.path.join(seg_test_path,img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(150, 150))
        X_test.append(np.array(img))



model = ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(150, 150, 3)), pooling='avg')    
print("\Testing image features are being prepared...")
prepare_data.convert_deep_features(X_test, model, prep_input_resnet50, 'resnet50_test.npz')       
   
X_test = np.load('resnet50_test.npz')['arr_0']
        
#---------------

loaded_model = pickle.load(open(model_file, 'rb'))

#---------------

if target_values != None:
    test_gt_df = pd.read_csv('test_label.csv', sep=";")
    test_gt = np.array(test_gt_df.Category)
    test_preds = loaded_model.predict(X_test)
    accuracy = (np.sum((test_preds - test_gt) == 0) / len(test_preds)) * 100
    test_preds_df = pd.DataFrame({'Category':test_preds})
    test_preds_df.to_csv('predictions.csv', index=False)
    print("Test set accuracy is ", accuracy)
    
    

elif target_values == None:
    test_preds = loaded_model.predict(X_test)
    test_preds_df = pd.DataFrame({'Category':test_preds})
    test_preds_df.to_csv('predictions.csv', index=False)
    print(test_preds)














