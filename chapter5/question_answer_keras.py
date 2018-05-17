#Machine Translation Demo 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, json
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, RepeatVector, TimeDistributed, Dense
from nltk.tokenize import word_tokenize
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


output_dim = 30; n_units = 300; activation = 'relu'
text_len = _text_len = 100; vocab_size = 1; input_shape = 50


def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])

def load_data():
    dataset = json.load(open('/Users/tawehbeysolow/Downloads/qadataset.json', 'rb'))['data']
    questions, answers = [], []
    for j in range(0, len(dataset)):
        for k in range(0, len(dataset[j])):
            for i in range(0, len(dataset[j]['paragraphs'][k]['qas'])):
                questions.append(remove_non_ascii(dataset[j]['paragraphs'][k]['qas'][i]['question']))
                answers.append(remove_non_ascii(dataset[j]['paragraphs'][k]['qas'][i]['answers'][0]['text']))
                
                
    _questions, _answers = ' '.join([q for q in questions]), ' '.join([a for a in answers])
    vocabulary = np.unique(word_tokenize(_questions + _answers))
    vocab_dictionary = {word: i for i, word in enumerate(vocabulary)}
    label_dictionary = {i: word for i, word in enumerate(vocabulary)}
                        
            
    
                        
    x = [[vocab_dictionary[word[0]] for word in sent] for sent in sentences]    
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[word[2]] for word in sent] for sent in sentences]  
    y = pad_sequences(maxlen=input_shape, sequences=y, padding='post', value=0)
    y = [np_utils.to_categorical(label, num_classes=len(label_dictionary)) for label in y]            
    
    return x, y, label_dictionary


def encoder_decoder():
    encoder = Input(shape=(text_len, ))
    encoder = Embedding(input_shape=vocab_size, output_dim=output_dim)(encoder)
    encoder = LSTM(n_units, return_sequences=True, activation=activation)(encoder)
    encoder = RepeatVector(_text_len)(encoder)    
    decoder = LSTM(n_units, return_sequences=True)(encoder)
    decoder = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder)
    model = Model(encoder, decoder)
    model.compile(loss='ctegorical_crossentropy', optimizer='adam',  metrics=['accuracy'])
    model.summary()
    return model
    

    