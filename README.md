# assignment3-2021-ardahuseyinoglu
assignment3-2021-ardahuseyinoglu created by GitHub Classroom

### HOW TO RUN:

#### *Training the model*:

`python train.py -train_data_path \data\seg_train -valid_data_path \data\seg_dev -img_feature resized_gray`

Args:
- train_data_path : Path of training dataset
- valid_data_path : Path of validation dataset
- img_feature : Use "resized_gray" or "deep_resnet50". 

Output:
- stores the learned model as .sav file.
- stores the extracted features (for train and validation set.)

Note: Best model is provided as "deep_resnet50". Although you will get far better accuracy rates, 
image features are extracted around 25 mins when using resnet50 features. So, that's the reason why "resized_gray" is given. 
Feature extraction lasts in seconds when using "resized_gray" and you can quickly get loss and accuracy changes to observe 
if the model learns without spending too much time in feature extraction.

------------

#### *Testing the best model*:

`python test.py -test_data_path \data\seg_test -model_file model.sav -target_values test.label.csv`

Args:
- test_data_path : Path of testing data
- model_file : Object of the best model which is stored in .sav format. This file must be in the same directory with test.py.
- target_values: Optional argument. If testing data has target values, you can provide a .csv file to get accuracy with predictions. 
                 If target values are not available, then do not use the argument (You will only have predictions for given test samples).
                 This file must be in the same directory with test.py. 

Output:
- stores the predictions for test dataset as .csv file.
- prints accuracy rate (if target values are provided.)

-----------------------------------------------------------------------------------------------------------------------------------------

***NOTE***
1. All works done for this assignment is in this current directory which includes "nn_model.py", "prepare_data.py", "train.py" and "test.py". 
However, "general" directory includes unstructured/unarranged code with other codes which provides plots, tables, images etc.
So you can just ignore the "general" directory. It's only to show how I get report items such as plots, tables, images.

