#Demo Theano Code
import theano
from theano import tensor as T
import numpy as np
from mnist import MNIST

#Functions and parameters to be used later
mndata = MNIST('/Users/tawehbeysolow/Downloads')
weight_shape = (784, 10)

def float_x(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(float_x(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

def load_data():
    '''
    Preprocessing data before inputting to neural network 
    :return: tuple
    '''
    train_x_data, train_y_data = mndata.load_training()
    test_x_data, test_y_data = mndata.load_testing()
    
    train_x, train_y = list(), list()
    test_x, test_y = list(), list()
    
    for i in range(0, len(train_x_data)):
        
    
    
def model_predict():
    train_x_data, train_y_data = mndata.load_training()
    test_x_data, test_y_data = mndata.load_testing()
    
    train_x, train_y = list(), list()
    test_x, test_y = list(), list()
    
    X = T.fmatrix()
    Y = T.fmatrix()
    
    weights = init_weights(weight_shape)
    
    predicted_y = model(X, weights)
    predicted_y = T.argmax(predicted_y, axis=1)
    
    cost = T.mean(T.nnet.categorical_crossentropy(predicted_y, Y))
    gradient = T.grad(cost=cost, wrt=weights)
    update = [[weights, weights - gradient * 0.05]]
    
    train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=predicted_y, allow_input_downcast=True)
    
    for i in range(100):
        for start, end in zip(range(0, len(train_x), 128), range(128, len(train_x), 128)):
            cost = train(train_x[start:end], train_y[start:end])
        print i, np.mean(np.argmax(test_y, axis=1) == predict(test_x))

if __name__ == '__main__':
    
    model_predict()