#RNN Tensorflow Demo
#Taweh Beysolow II

#Import necessary modules
import numpy as np
import tensorflow as tf

#Setting parameters to be used
learning_rate = 0.02
epochs = 600 #Total number of epochs
series_len = 50000 # Length of all of the sequences
bprop_len = 2 #Length of indices for batch processing
state_size = 4
n_classes = 2 #Number of classes for data observations
batch_size = 200
n_batches = series_len//batch_size//bprop_len #Total number of batches to iterate through

#Generating data for the RNN
def generate_data():
    x = np.random.choice(2, series_len, p = [0.5, 0.5]) #Creating Binary Vector with equal probability of classes, of length 50000
    y = np.roll(x, 3) #Creating Y variable by shifting the X variable by 3 places
    y[0:3] = 0 #Changing first 4 observations to 0
    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))
    return (x, y)

#Vanilla RNN Implementation
def build_rnn(learning_rate=0.02, epochs=600, state_size=4):
    
    #Loading data 
    x, y = generate_data()
    #train_x, train_y = x[0:int(np.floor(len(x)*.67)), :], y[0:int(np.floor(len(x)*.67))]
    #test_x, test_y = x[int(np.floor(len(x)*.67)):, ], y[int(np.floor(len(x)*.67))]                    
    
    #Creating weights and biases dictionaries
    weights = {'input': tf.Variable(tf.random_normal([state_size+1, state_size])),
        'output': tf.Variable(tf.random_normal([state_size, n_classes]))}
    biases = {'input': tf.Variable(tf.random_normal([1, state_size])),
        'output': tf.Variable(tf.random_normal([1, n_classes]))}

    #Defining placeholders and variables
    X = tf.placeholder(tf.float32, [batch_size, bprop_len])
    Y = tf.placeholder(tf.int32, [batch_size, bprop_len])
    init_state = tf.placeholder(tf.float32, [batch_size, state_size])
    input_series = tf.unstack(X, axis=1)
    labels = tf.unstack(Y, axis=1)
    current_state = init_state
    hidden_states = []

    #Passing values from one hidden state to the next
    for input in input_series: #Evaluating each input within the series of inputs
        input = tf.reshape(input, [batch_size, 1]) #Reshaping input into MxN tensor
        input_state = tf.concat([input, current_state], axis=1) #Concatenating input and current state tensors
        _hidden_state = tf.tanh(tf.add(tf.matmul(input_state, weights['input']), biases['input'])) #Tanh transformation
        hidden_states.append(_hidden_state) #Appending the next state
        current_state = _hidden_state #Updating the current state

    logits = [tf.add(tf.matmul(state, weights['output']), biases['output']) for state in hidden_states]
    predicted_labels = [tf.nn.softmax(logit) for logit in logits] #predictions for each logit within the series

    #Creating error, accuracy, and optimizer variables
    error = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in zip(logits, labels)]
    cross_entropy = tf.reduce_mean(error)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(predicted_labels, 1)), tf.float32))

    #Execute Graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #Loading data from function
        for epoch in range(epochs):
  
            _state = np.zeros([batch_size, state_size])

            #Creating batches for training
            for batch in range(n_batches):
                start = batch*bprop_len
                end = start+bprop_len

                batch_x = x[:, start:end]
                batch_y = y[:, start:end]

                #Evaluating Network Performance
                _error, _error, _state, _accuracy = sess.run([optimizer, cross_entropy, init_state, accuracy],
                                                                  feed_dict={X:batch_x, Y:batch_y, init_state:_state})
                
            #Printing logging information on network
            if epoch%20 == 0:
                print('Epoch: ' + str(epoch) +  
                '\nError:' + str(_error) + 
                '\nAccuracy: ' + str(_accuracy) + '\n')
            
        #predicted_test_labels = sess.run(predicted_labels, feed_dict={X: x})
        #test_error = sess.run(cross_entropy, feed_dict={X:x, Y: y})
        #print('Test Error: ' + str(test_error))
        #print(predicted_test_labels)
        

if __name__ == '__main__': 
    
    build_rnn()
