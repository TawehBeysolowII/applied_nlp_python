#LSTM Tensorflow Demo 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn
import pandas as pan 
from sklearn.preprocessing import MinMaxScaler
import math

#Parameters
input_shape=(1, 5)
n_units=300
batch_size=5
learning_rate=1e-3
training_steps=1000
epochs=100
n_classes=2
series_len=10000
state_size=4
n_hidden=n_units

#Generating data for the RNN
def generate_data():
    x = np.random.choice(2, series_len, p=[0.5, 0.5]) #Creating Binary Vector with equal probability of classes, of length 50000
    y = np.roll(x, 3) #Creating Y variable by shifting the X variable by 3 places
    y[0:3] = 0 #Changing first 4 observations to 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x, y)

def train_lstm(learning_rate=learning_rate, n_units=n_units, epochs=epochs):

    x, y = generate_data()
    train_x, train_y = x[0:int(math.floor(len(x)*.67)),  :], y[0:int(math.floor(len(y)*.67))]
  
    #Defining weights in addition to placeholder variables     
    weights = {'output': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'output': tf.Variable(tf.random_normal([n_classes]))}      
    X, Y = tf.placeholder(tf.int32, shape=(None, None, 1)), tf.placeholder(tf.int32, shape=(None, None, n_classes))
    X = tf.reshape(X, [-1, input_shape[1]])
    X = tf.split(X, input_shape[1], 1)

    lstm = rnn.BasicLSTMCell(num_units=n_units, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm, inputs=X, sequence_length=batch_size, dtype=tf.int32)
    softmax_output = tf.nn.softmax(tf.add(tf.matmul(outputs, weights['output']), biases['output']))
    
    accuracy = tf.equal(tf.argmax(softmax_output, 1), tf.argmax(Y, 1))
    error = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=softmax_output)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initiliazer())
        
        for epoch in range(epochs):

            
            for start, end in zip(range(0, len(train_x)-32, 32), range(32, len(train_x), 32)):
                
                _train_x, _train_y = train_x[start:end], train_y[start:end]
                _error, _accuracy = sess.run([adam_optimizer, accuracy],  feed_dict={X: _train_x, Y: _train_y})
                
                if epoch%10 == 0:
                    print('Epoch: ' + str(epoch) + 
                    '\n Accuracy: ' +  "{.4f}".format(_accuracy) + 
                    '\n Error: ' + "{.4f}".format(_error))
                    
if __name__ == '__main__':
    
    train_lstm()