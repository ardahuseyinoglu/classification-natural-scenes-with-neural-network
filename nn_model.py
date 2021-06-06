import numpy as np
from tqdm import tqdm 

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
    expo_sum = np.sum(np.exp(x), axis=1).reshape(-1,1)
    return expo/expo_sum