#### Importing Libraries
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

# manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2 
import os
from tqdm import tqdm 
import pickle



#### Loading image features (X) & target values (Y)
X_train_ = np.load('resnet50_train.npz')['arr_0']
X_valid_ = np.load('resnet50_valid.npz')['arr_0']
X_test_ = np.load('resnet50_test.npz')['arr_0']

Y_train_encoded = np.load('Y_train_encoded.npz')['arr_0']
Y_valid_encoded = np.load('Y_valid_encoded.npz')['arr_0']


#### Shuffling
random_index = np.arange(len(X_train_))
np.random.shuffle(random_index)
X_train_ = X_train_[random_index]
Y_train_encoded = Y_train_encoded[random_index]

random_index = np.arange(len(X_valid_))
np.random.shuffle(random_index)
X_valid_ = X_valid_[random_index]
Y_valid_encoded = Y_valid_encoded[random_index]


#### Model
class Layer:
    def __init__(self, input_size, output_size, activation_name):
        self.input = None
        self.output = None
        self.Z = None
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.activation_name = activation_name
        
        if activation_name == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        elif activation_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation_name == 'relu':
            self.activation = relu
            self.activation_deriv = relu_deriv
        elif activation_name == 'l_relu':
            self.activation = l_relu
            self.activation_deriv = l_relu_deriv
        elif activation_name == 'softmax':
            self.activation = softmax
            self.activation_deriv = None
        else:
            self.activation = None
            self.activation_deriv = None


    def forward_propagation(self, input_data):
        self.input = input_data
        self.Z = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.Z)
        return self.output


    def backward_propagation(self, output_error, learning_rate):
        
        if self.activation_name == 'softmax':
            dZ = output_error
        else:
            dZ = self.activation_deriv(self.Z) * output_error
            
        dX = np.dot(dZ, self.weights.T)
        dW = np.dot(self.input.T, dZ)
        dB = np.sum(dZ, axis=0).reshape(1,-1)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * dB
        return dX




class NeuralNetwork:
    def __init__(self, loss_name):
        self.layers = []
        self.loss_name = loss_name
        
        if loss_name == 'mse':
            self.loss = mse
            self.loss_deriv = mse_deriv
        
        elif loss_name == 'cross_entropy':
            self.loss = cross_entropy
            self.loss_deriv = cross_entropy_deriv
        
        else:
            self.loss = None
            self.loss_deriv = None
        

    def add(self, layer):
        self.layers.append(layer)

        
    def predict(self, input_data):
        output = input_data[:]
        
        for layer in self.layers:
            output = layer.forward_propagation(output)
        
        output = np.argmax(output, axis=1)
        
        return output
    
    
    def get_accuracy(self, x, y):
        preds = self.predict(x)
        gt = np.argmax(y, axis=1)
        accuracy = np.sum((preds - gt) == 0) / len(preds)
        
        return accuracy*100

    
    def train(self, x_train, y_train, x_valid, y_valid, epochs, batch_size, learning_rate, lr_decay_params):
        
        init_lr = learning_rate
        
        training_accs = []
        validation_accs = []
        training_losses = []
        validation_losses = []
        
        num_training_samples = len(x_train)
        num_validation_samples = len(x_valid)
        num_iter_per_epoch = np.ceil(num_training_samples / batch_size)

        for epoch in range(epochs):
            print('Epoch ', str(epoch+1), '/', str(epochs), ':')
            
            learning_rate = init_lr * np.power(lr_decay_params['drop_amount'], np.floor((1+epoch)/lr_decay_params['num_epochs_to_drop']))

            training_epoch_loss = 0
            validation_epoch_loss = 0
            
            output = x_valid[:]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            
            validation_epoch_loss += self.loss(y_valid, output)
            
            
            for training_sample_index in tqdm(range(0, num_training_samples, batch_size)):
                output = x_train[training_sample_index:training_sample_index + batch_size]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                training_epoch_loss += self.loss(y_train[training_sample_index:training_sample_index + batch_size], output)
                
                error = self.loss_deriv(y_train[training_sample_index:training_sample_index + batch_size], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
    

            # calculate average error on all samples
            training_epoch_loss /= num_iter_per_epoch
            training_losses.append(training_epoch_loss)
            validation_losses.append(validation_epoch_loss)
            
            epoch_acc_train = self.get_accuracy(x_train, y_train)
            epoch_acc_valid = self.get_accuracy(x_valid, y_valid)
            training_accs.append(epoch_acc_train)
            validation_accs.append(epoch_acc_valid)
            
            print('Training loss: ', str(training_epoch_loss), '\tTraining accuracy: ', str(epoch_acc_train))
            print('Validation loss: ', str(validation_epoch_loss), '\tValidation accuracy: ', str(epoch_acc_valid))
            print("---------------------------------------------------------------------\n")
            
        return training_accs, validation_accs, training_losses, validation_losses



def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def relu(x):
    return np.maximum(0,x)

def relu_deriv(x):
    return 1 * (x >= 0)

def l_relu(x):
    return np.maximum(0.01*x, x)

def l_relu_deriv(x):
    x[x>=0] = 1
    x[x<0] = 0.01
    return x

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_deriv(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(y_true, y_pred):
    # to avoid divided by zero error
    epsilon = 1e-15
    return np.mean(np.sum(-1 * (np.log(y_pred + epsilon) * y_true), axis=1))

def cross_entropy_deriv(y_true, y_pred):
    return (y_pred - y_true)/y_true.size

def softmax(x):
    expo = np.exp(x)
    expo_sum = np.sum(np.exp(normalized_x), axis=1).reshape(-1,1)
    return expo/expo_sum




# training/validation data
x_train = X_train_[:]
y_train = Y_train_encoded[:]
x_valid = X_valid_[:]
y_valid = Y_valid_encoded[:]

# network
model = NeuralNetwork('mse')
model.add(Layer(2048,512, 'tanh'))
model.add(Layer(512,128, 'tanh'))
model.add(Layer(128,64, 'tanh'))
model.add(Layer(64,6, 'tanh'))

# hyperparams
num_epochs = 50
batch_size = 32
lr = 0.1
lr_decay_params = {'drop_amount':0.75, 
                   'num_epochs_to_drop': 10}

# train
training_accs, validation_accs, training_losses, validation_losses = model.train(x_train, 
                                                                               y_train, 
                                                                               x_valid, 
                                                                               y_valid, 
                                                                               epochs = num_epochs, 
                                                                               batch_size = batch_size, 
                                                                               learning_rate = lr, 
                                                                               lr_decay_params = lr_decay_params)


##### Testing

test_gt_df = pd.read_csv('test_label.csv', sep=";")
test_gt = np.array(test_gt_df.Category)

test_preds = model.predict(X_test_)

accuracy = (np.sum((test_preds - test_gt) == 0) / len(test_preds)) * 100
accuracy


##### Plot accuracy and loss for training/validation

# accuracy 
plt.figure(figsize=(18,4))
num_skip = 5
acc_plot = sns.lineplot(x=np.arange(1, len(training_accs)+1) , y=training_accs, label='Training')
acc_plot = sns.lineplot(x=np.arange(1, len(validation_accs)+1) , y=validation_accs, label='Validation')
acc_plot.set(xlabel='Epoch', ylabel='Accuarcy (%)')
#acc_plot.set(ylim=(0, 105))
plt.xticks(np.arange(0, len(training_accs)+1, num_skip))
plt.show()


# loss
plt.figure(figsize=(18,4))
num_skip = 5
acc_plot = sns.lineplot(x=np.arange(1, len(training_losses)+1) , y=training_losses, label='Training')
acc_plot = sns.lineplot(x=np.arange(1, len(validation_losses)+1) , y=validation_losses, label='Validation')
acc_plot.set(xlabel='Epoch', ylabel='Loss')
#acc_plot.set(ylim=(0, 0.3))
plt.xticks(np.arange(0, len(training_accs)+1, num_skip))
plt.show()


##### Save and load model

pickle.dump(model, open('model_no.sav', 'wb'))
loaded_model = pickle.load(open('model3.sav', 'rb'))

# loaded_model.get_accuracy(X_valid_, Y_valid_encoded)
# or just get prediction for given dataset
# loaded_model.predict(x_test)


##### Visualization of Parameters (using model5 architecture and hyperparameters)
fig,ax=plt.subplots(2,3)
fig.set_size_inches(10,10)

ax[0,0].imshow(loaded_model.layers[1].weights[:,0].reshape(30,30),cmap='gray')
ax[0,0].set_title("Buildings (o_0)")
ax[0,0].grid(None)

ax[0,1].imshow(loaded_model.layers[1].weights[:,1].reshape(30,30),cmap='gray')
ax[0,1].set_title("Forest (o_1)")
ax[0,1].grid(None)

ax[0,2].imshow(loaded_model.layers[1].weights[:,2].reshape(30,30),cmap='gray')
ax[0,2].set_title("Glacier (o_2)")
ax[0,2].grid(None)

ax[1,0].imshow(loaded_model.layers[1].weights[:,3].reshape(30,30),cmap='gray')
ax[1,0].set_title("Mountain (o_3)")
ax[1,0].grid(None)

ax[1,1].imshow(loaded_model.layers[1].weights[:,4].reshape(30,30),cmap='gray')
ax[1,1].set_title("Sea (o_4)")
ax[1,1].grid(None)

ax[1,2].imshow(loaded_model.layers[1].weights[:,5].reshape(30,30),cmap='gray')
ax[1,2].set_title("Street (o_5)")
ax[1,2].grid(None)


##### Confusion matrix and classification report
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
sns.heatmap(confusion_matrix(test_gt, test_preds), 
            annot=True, 
            fmt="d", 
            cmap="YlGnBu",
            xticklabels=classes, 
            yticklabels=classes)

plt.show()
print(classification_report(y_ground_truth, preds, target_names=classes))


##### Misclassified Samples
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

seg_dev_path = os.path.join('data','seg_dev')
X_valid, Y_valid, number_of_samples_valid = load_data(path=seg_dev_path)


print(np.where((preds - y_ground_truth) != 0))

def show_misclassifies_sample(i):
    print("ground truth: ", classes[y_ground_truth[i]])
    print("prediction: ", classes[preds[i]])
    plt.imshow(X_valid[random_index[i]])
    plt.grid(None)
    plt.show()

show_misclassifies_sample(667)


##### Kaggle

test_gt_df = pd.read_csv('test_label.csv', sep=";")
test_gt = np.array(test_gt_df.Category)

test_preds = loaded_model.predict(X_test_)

accuracy = (np.sum((test_preds - test_gt) == 0) / len(test_preds)) * 100
print(accuracy)

test_preds_df = pd.DataFrame({'Id': test_gt_df.Id.values, 'Category':test_preds})

preds_df.to_csv('submission_ah.csv', index=False)

df = pd.read_csv("submission_ah.csv")


