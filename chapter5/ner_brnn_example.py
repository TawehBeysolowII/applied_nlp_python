#Name entity recognition example with trainer
#Taweh Beysolow II

#Import the necessary modules
import numpy as np, math, string, pandas as pan
from keras.utils import np_utils
from keras.models import Model, Input
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed
from keras.preprocessing.sequence import pad_sequences

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
input_shape = 75
epochs = 10

def load_data():
    text_data = open('/Users/tawehbeysolow/Downloads/train.txt', 'rb').readlines()
    text_data = [text_data[k].replace('\t', ' ').split() for k in range(0, len(text_data))]
    index = range(0, len(text_data), 3)
    
    #Transforming data to matrix format for neural network
    input_data =  list()
    for i in range(1, len(index)-1):
        rows = text_data[index[i-1]:index[i]]
        sentence_no = np.array([i for i in np.repeat(i, len(rows[0]))], dtype=str)
        rows.append(np.array(sentence_no))
        rows = np.array(rows).T
        input_data.append(rows)
    
    input_data = pan.DataFrame(np.concatenate([input_data[j] for j in range(0,len(input_data))]), 
                           dtype=str,
                           columns=['word', 'pos', 'tag', 'sent_no'])
    
    labels, vocabulary = list(set(input_data['tag'].values)), list(set(input_data['word'].values))
    vocabulary.append('endpad'); vocab_size = len(vocabulary); label_size = len(labels)
    
    agg_func = lambda input: [(w, p, t) for w, p, t in zip(input['word'].values.tolist(),
                                                      input['pos'].values.tolist(),
                                                       input['tag'].values.tolist())]
                           
    grouped_input_data= input_data.groupby('sent_no').apply(agg_func)
    sentences = [s for s in grouped_input_data]    
    word_dictionary = {word: i for i, word in enumerate(vocabulary)}
    label_dictionary = {label: i for i, label in enumerate(labels)}
    x = [[word_dictionary[word[0]] for word in sent] for sent in sentences]    
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[word[2]] for word in sent] for sent in sentences]  
    y = pad_sequences(maxlen=input_shape, sequences=y, padding='post', value=0)
    y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]            
    return x, y, label_dictionary, vocab_size, label_size
    
def train_lstm_keras():
    
    x, y, label_dictionary, vocab_size, label_size =  load_data()
    train_end = int(math.floor(len(x)*.67))
    train_x, train_y = x[0:train_end] , np.array(y[0:train_end])
    test_x, test_y = x[train_end:] , np.array(y[train_end:])

    def create_lstm():        
        input = Input(shape=(input_shape, ))
        model = Embedding(input_dim=vocab_size+1, output_dim=20,
                          input_length=input_shape, mask_zero=True)(input)
        
        model = Bidirectional(LSTM(units=n_units, activation=activation, 
                                   return_sequences=True))(model)
        output = TimeDistributed(Dense(label_size, activation=out_act))(model) 
        model = Model(input, output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
            
    lstm_model = create_lstm()
    lstm_model.fit(train_x, train_y, epochs=epochs, validation_split=0.10, 
                   shuffle=True, batch_size=batch_size)

if __name__ == '__main__':
    
    train_lstm_keras()
    
