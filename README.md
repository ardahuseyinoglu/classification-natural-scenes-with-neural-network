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

-----------------------------------------------------------------------------------------------------------------------------------------

# Classification of Natural Scenes using Neural Network

In this assignment, we will implement a single and multi layer neural network architecture from scratch to classify the image scenes into 6 classses which are buildings, forest, glacier, mountain, sea and street. We will analyze the effects of changing the number of hidden layers and number of hidden units. We will observe the effect of other hyperparameters such as learning rate, batch size, number of epochs and learning rate decay. In experiments, we will also use different activation functions for hidden layers and use different cost functions.

<br>
<p align="center">
  <b>Some sample images from training set</b>
</p>

<p align="center">
  <img src="/report-images/samples.PNG">
</p>

<br>
<p align="center">
  <b>Training set sample distribution</b>
</p>

<p align="center">
  <img src="/report-images/train_dist.PNG">
</p>


To visualize the learned parameters, we use a neural network architecture with one hidden layer $(2048 \rightarrow 900 \rightarrow 6)$. We choose the hidden unit size as 900 to plot the weights properly in 30x30 images. The reason why the images below does not make much sense may be that we use ResNet-50 architecture to exract image features and these transformed features does not represent the images as we see. However, if we would use CNN architecture, then visualization of filters would make sense. Because filters in layers learn how to represent the input image.

We first try using different image representation methods. Firstly, we use a traditional method that is resizing the images to 30x30 and convert them into grayscale. After that, we use different CNN architectures which is pre-trained on 'Imagenet' dataset , such as VGG-16, VGG-19, ResNet-50, InceptionV3, to extract image features. Pre-trained filter weights surely increase our accuracy in comparison with the traditional method. Because we use filters in CNN models that knows how to represent an image. The representation power of filters are pretty good since we use models trained on Imagenet dataset, which have many classes and have everyday object/scenes. In addition, since we classify the scenes on the images, we may need to use color information. So, when we convert the images into the gray scale, we lose it. The model can use it in many ways. For example, green-colored pixels assists to model to make the prediction as 'forest'. Since we use raw input image with CNNs, we also benefit the color information of images.

We get the best accuracy rate of 94.0%, when we use the ResNet-50 architecture.

<p align="center">
  <img src="/report-images/accs1.PNG">
</p>


After deciding the image representation method, we decide which neureal network architecture we will use for classification. We try different numbers of layers and different numbers of hidden units for each layer. When we use less number of hidden units and use less number of hidden layers, our model tends to underfit in comparison with the more complex models. We get the best accuracy rate of 94.36%, when we use 3 hidden layers with the hidden unit sizes of 2048, 512, 128, 64, 6, respectively.

<p align="center">
  <img src="/report-images/accs2.PNG">
</p>


After we structure the model, we tune the hyperparameters to improve our results. Decreasing the learning rate turns out to decrease the accuracy rates, since the model needs more number of epoch to converge. So we decide to keep learning rate as 0.1 (since we will use learning rate decay and it reduces the learning rate in next epochs, learning rate value of 0.1 will be a good starting point). When the batch size increases, the model learns faster since vectorization is used more. However, we cannot keep all the data once in memory. So we need to decide a batch size which provides optimal generalization. After trying diffent sizes such as 8,16,32,64,128 and 256, we get the best accuracy rate of 94.30 for the batch size of 32.


Learning rate decay parameters has also play an important role. Since we need to use smaller learning rates to converge better when we come near to the minima point, we should decrease the learning rate for next epochs. We get the best accuracy rates when we multiply the learning rate with 0.75 after each 10 epoch.


Our final model is a multi-layer neural network with 3 hidden layers with learning rate of 0.1, batch size of 32, number of epochs of 50, MSE loss function and tanh activation function for hidden layers

<p align="center">
  <img src="/report-images/accs3.PNG">
</p>


The model generally misclassifies the scenes that includes some of the visual elements belonging to the predicted class. For example, images that include snow-capped mountains are predicted as glacier. Also, building images that include streets or street images that includes some buildings are misclassified. The two highest f1-score belongs to classes which are forest and sea. The reason why is that they can be easily differentiated from other classes. Building and street labeled images have similar texture, shapes and colors like glacier and mountain labeled images. However, content of the forest and sea labeled images are unique according to the other 4 classes and


<br>
<p align="center">
  <b>Confusion matrix and classification report for test set</b>
</p>

<p align="center">
  <img src="/report-images/conf_mat.PNG">
</p>

Also, there are some misclassified images whose ground truth label could be labeled as predicted class, since the visual elements of the misclassified image can be interpreted by also that way. In other words, the model has difficulty in predicting the environments that are intertwined with each other. For example, mountains, sea or forests generally can be seen in the same image, or there can be scenes includes road and mountain, or most of the street images includes also buildings and vice versa. The fact that these photos, which can have more than one tag, belong to a single class, decrease the accuracy rate of the model. We can see that the second images at third and fourth row below have almost the same content but labeled differently.

<br>
<p align="center">
  <b>Some misclassified images</b>
</p>

<p align="center">
  <img src="/report-images/misclassified.PNG">
</p>

<p align="center">
  <img src="/report-images/misclass_interesting.PNG">
</p>
