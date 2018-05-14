#Name entity recognition example with trainer
#Taweh Beysolow II

#Import the necessary modules
import tensorflow as tf, numpy as np, math, string
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize

#Parameters
np.random.seed(2018)
learning_rate = 0.01
momentum = 0.9
activation = 'selu'
out_act = 'softmax'
opt = 'adam'
n_units = 1000
batch_size = 150
punctuation= set(string.punctuation)

def create_dictionary(data, inverse=False):
    output = {}
    if inverse == False:
        for i in range(0, len(data)):
            output[data[i]] = i
    else:
        for i in range(0, len(data)):
            output[i] = data[i]
    return output
    
def label_transform(labels, dictionary):
    for k in range(0, len(labels)):
        for i in range(0, len(labels[k])):
            labels[k][i] = dictionary[labels[k][i]]
    return labels

def preprocess_data(window_size=2, skip_gram=True):
    
    text_data = open('/Users/tawehbeysolow/Downloads/train.txt', 'rb').readlines()
    text_data = [text_data[k].replace('\t', ' ').split() for k in range(0, len(text_data))]
    input_index, output_index = range(0, len(text_data), 3), range(1, len(text_data), 3)
    x = list([text_data[index] for index in input_index])
    y = list([text_data[index] for index in output_index])
    y = [y[k][len(y[k])-1] for k in range(0, len(y))]
    concat_data = np.concatenate([text_data[k] for k in range(0, len(text_data))])
    concat_y, concat_data = np.unique(y), np.unique(concat_data)
    word_dictionary = create_dictionary(data=concat_data)
    index_dictionary = create_dictionary(data=concat_y, inverse=True)
        
    def one_hot_encoder(indices, vocab_size, skip_gram):
        vector = np.zeros(vocab_size)
        if skip_gram == True: vector[indices] = 1
        else:
            for index in indices: vector[index] = 1  
        return vector
        
    vocab_size, n_gram_data = len(concat_data), []

    for k in range(0, len(x)): #Creating word pairs for skip_gram model
        for index, word in enumerate(x[k]):
            if word not in punctuation: #Removing grammatical objects from input data
                for _word in x[k][max(index - window_size, 0): min(index + window_size, len(x[k])) + 1]:
                    if _word != word: #Making sure not to duplicate word_1 when creating n-gram lists
                        n_gram_data.append([word, _word])
    
    x, y = np.zeros([len(n_gram_data), vocab_size]), np.zeros([len(n_gram_data), vocab_size])   
    for i in range(0, len(n_gram_data)): #Concatenating one-hot encoded vector into input and output matrices
        x[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][0]], vocab_size=vocab_size, skip_gram=skip_gram)      
        y[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][1]], vocab_size=vocab_size, skip_gram=skip_gram)            

    return x, y, vocab_size, index_dictionary


def load_data():
    text_data = open('/Users/tawehbeysolow/Downloads/train.txt', 'rb').readlines()
    text_data = [text_data[k].replace('\t', ' ').split() for k in range(0, len(text_data))]
    input_index, output_index = range(0, len(text_data), 3), range(1, len(text_data), 3)
    x = list([text_data[index] for index in input_index])
    y = list([text_data[index] for index in output_index])
    y = [y[k][len(y[k])-1] for k in range(0, len(y))]
    concat_data = np.concatenate([text_data[k] for k in range(0, len(text_data))])
    concat_y, concat_data = np.unique(y), np.unique(concat_data)
    
    #Transforming labels
    def create_dictionary(data, inverse=False):
        output = {}
        if inverse == False:
            for i in range(0, len(data)):
                output[data[i]] = i
        else:
            for i in range(0, len(data)):
                output[i] = data[i]
        return output
        
    def label_transform(labels, dictionary):
        for k in range(0, len(labels)):
            for i in range(0, len(labels[k])):
                labels[k][i] = dictionary[labels[k][i]]
        return labels

    word_dictionary = create_dictionary(data=concat_data)
    label_dictionary = create_dictionary(data=concat_y)
    index_dictionary = create_dictionary(data=concat_y, inverse=True)
    x = label_transform(labels=x, dictionary=word_dictionary)     
    y = [label_dictionary[y[k]] for k in range(0, len(y))]
    return x, y, index_dictionary
    
def train_lstm_keras():
    
    x, y, vocab_size, index_dictionary = preprocess_data()
    x, y, index_dictionary = load_data()
    encoder = OneHotEncoder()
    x = encoder.fit_transform(x)
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end] , np_utils.to_categorical(y[0:train_end])
    test_x, test_y = x[train_end:] , y[train_end:]
    train_x  = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
    test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])
    
    def create_lstm(input_shape=(1, x.shape[1])):
        model = Sequential()
        model.add(Bidirectional(LSTM(unites=n_units, 
                                     activation=activation),
                                     input_shape=input_shape))

        model.add(Dense(train_y.shape[1]), activation=out_act)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    
    lstm_model = create_lstm()
    lstm_model.summary()
    lstm_model.fit(train_x, train_y, shuffle=True, batch_size=batch_size)

    

if __name__ == '__main__':
    
    train_lstm_keras()
    
