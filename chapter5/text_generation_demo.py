#Text Generation Demo
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np
from chapter4.word_embeddings import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, Dense

max_pages = 10
pdf_file = 'harry_potter.pdf'
misc = '''... '' -- '''.split()
sequence_length = 50

def preprocess_data(sequnece_length=sequence_length, pdf_file=pdf_file, max_pages=max_pages):
    x, y, character_dict = [], [], {}
    text_data = load_data(raw_text=True, pdf_file=pdf_file, max_pages=max_pages)
    unique_chars = sorted(list(set(text_data.lower())))
    character_dict = dict((c, i) for i, c in enumerate(unique_chars))
    num_chars, vocab_length = len(text_data), len(text_data.split())    
    
    for i in range(0, len(text_data) - sequnece_length, 1):
        input_sequence = text_data[i: i+sequence_length]
        output_sequence = text_data[i+sequence_length]
        x.append(character_dict[char.lower()] for char in input_sequence)
        y.append(character_dict[output_sequence.lower()])
        
    for i in range(0, len(x)): #Transforming from generators
        x[i] = [_x for _x in x[i]]

    x = np.reshape(x, (len(x), sequence_length, 1))
    x, y = x/float(vocab_length), np_utils.to_categorical(y)
    return x, y, num_chars, vocab_length 
    
def train_rnn_keras(epochs, batch_size, activation, num_units):

    x, y, num_chars, vocab_length = preprocess_data()
    
    def create_rnn(num_units=num_units, activation=activation):
        model = Sequential()
        model.add(LSTM(num_units, activation=activation, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
        model.add(LSTM(num_units, activation=activation))
        model.add(Dense(y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])            
        model.summary()
        return model
            
    rnn_model = create_rnn()
    rnn_model.fit(x, y, epochs=epochs, batch_size=batch_size)
   
if __name__ == '__main__':
    
    train_rnn_keras(epochs=10, batch_size=100, num_units=200, activation='relu')