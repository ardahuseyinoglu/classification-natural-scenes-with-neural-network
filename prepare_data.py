import numpy as np
import cv2 
from tqdm import tqdm 
import os

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



###### Resized & Gray Scale
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