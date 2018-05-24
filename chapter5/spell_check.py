#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#Spell Check via BRNN 
#Taweh Beysolow II 

#Import the necessary packages 
import numpy as np, string 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from nltk.tokenize import word_tokenize, sent_tokenize

#Parameters
sequence_len = 10
punctuation = set(string.punctuation)
n_units = 300
epochs = 10
validation_split = 0.10
batch_size = 100
activation = 'relu'
    
def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])
            
def load_data(window_size=5, skip_gram=True):
    raw_text = open('/Users/tawehbeysolow/Downloads/large_corpus.txt', 'rb').readlines()[0:1000]
    text_data = ' '.join([remove_non_ascii(line) for line in raw_text])
            
    def one_hot_encoder(indices, vocab_size, skip_gram):
        vector = np.zeros(vocab_size)
        if skip_gram == True: vector[indices] = 1
        else:
            for index in indices: vector[index] = 1  
        return vector
        
    vocab_size, word_dictionary, index_dictionary, n_gram_data = len(word_tokenize(text_data)), {}, {},  []

    for index, word in enumerate(word_tokenize(text_data)):
        word_dictionary[word], index_dictionary[index] = index, word
           
    sentences = sent_tokenize(text_data) #Tokenizing sentences
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences]) #Creating lists of words for each tokenized setnece

    for sentence in tokenized_sentences: #Creating word pairs for skip_gram model
        for index, word in enumerate(sentence):
            if word not in punctuation: #Removing grammatical objects from input data
                for _word in sentence[max(index - window_size, 0): min(index + window_size, len(sentence)) + 1]:
                    if _word != word: #Making sure not to duplicate word_1 when creating n-gram lists
                        n_gram_data.append([word, _word])
    
    x, y = np.zeros([len(n_gram_data), vocab_size]), np.zeros([len(n_gram_data), vocab_size])
    
    for i in range(0, len(n_gram_data)): #Concatenating one-hot encoded vector into input and output matrices
        x[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][0]], vocab_size=vocab_size, skip_gram=skip_gram)      
        y[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][1]], vocab_size=vocab_size, skip_gram=skip_gram)            

    return x, y, vocab_size, index_dictionary

def train_spell_check():
    
    x, y, vocab_size, label_dictionary = load_data()
    x = x.reshape(x.shape[0], 1, x.shape[1])
    
    
    def create_brnn():
        model = Sequential()
        model.add(Bidirectional(LSTM(units=n_units, return_sequences=True),
                                input_shape=(None, vocab_size)))
        
        model.add(Bidirectional(LSTM(units=n_units, go_backwards=True)))
        model.add(Dense(vocab_size, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
        
    brnn_model = create_brnn()
    
    brnn_model.fit(x, y, batch_size=batch_size, 
                      validation_split=validation_split, 
                      epochs=epochs,
                      shuffle=True)
        
    return brnn_model

def spell_check_examples():
    
    load_test_data = open('/Users/tawehbeysolow/Downloads/mispelled_words.txt', 'rb')
    
if __name__ == '__main__': 
    
    train_spell_check()
    
    

            
    
    