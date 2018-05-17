#Machine Translation Demo 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, json
from keras.models import Model, Input
from keras.layers import LSTM, Dense

#Parameters
n_units = 300;
activation = 'relu'
epochs = 1

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
                
    input_chars, output_chars = set(), set()
    
    for i in range(0, len(questions)):
        for char in questions[i]: 
            if char not in input_chars: input_chars.add(char.lower())
    
    for i in range(0, len(answers)):
        for char in answers[i]:
            if char not in output_chars: output_chars.add(char.lower())
    
    input_chars, output_chars = sorted(list(input_chars)), sorted(list(output_chars))
    n_encoder_tokens, n_decoder_tokens = len(input_chars), len(output_chars)
    max_encoder_len = max([len(text) for text in questions])
    max_decoder_len = max([len(text) for text in answers])
    
    input_dictionary = {word: i for i, word in enumerate(input_chars)}
    output_dictionary = {word: i for i, word in enumerate(output_chars)}
    label_dictionary = {i: word for i, word in enumerate(output_chars)}
    
    x_encoder = np.zeros((len(questions), max_encoder_len, n_encoder_tokens))
    x_decoder = np.zeros((len(questions), max_decoder_len, n_decoder_tokens))
    y_decoder = np.zeros((len(questions), max_decoder_len, n_decoder_tokens))

    for i, (input, output) in enumerate(zip(questions, answers)):
        for _character, character in enumerate(input):
            x_encoder[i, _character, input_dictionary[character.lower()]] = 1.
    
        for _character, character in enumerate(output):
            x_decoder[i, _character, output_dictionary[character.lower()]] = 1.

            if i > 0: y_decoder[i, _character, output_dictionary[character.lower()]] = 1.

    data = list([x_encoder, x_decoder, y_decoder])      
    variables = list([label_dictionary, n_decoder_tokens, n_encoder_tokens])                             
    return data, variables


def encoder_decoder(n_encoder_tokens, n_decoder_tokens):
    
    encoder_input = Input(shape=(None, n_encoder_tokens))    
    encoder = LSTM(n_units, return_state=True, activation=activation)
    encoder_output, hidden_state, cell_state = encoder(encoder_input)
    encoder_states = [hidden_state, cell_state]
    
    decoder_input = Input(shape=(None, n_decoder_tokens))
    decoder = LSTM(n_units, return_state=True, return_sequences=True, activation=activation)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_states)
    
    decoder = Dense(n_decoder_tokens, activation='softmax')(decoder_output)
    model = Model([encoder_input, decoder_input], decoder)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    model.summary()
    return model

    
def train_encoder_decoder():
    
    input_data_objects = load_data()
    x_encoder, x_decoder, y_decoder = input_data_objects[0][0], input_data_objects[0][1], input_data_objects[0][2]
    output_dictionary, n_decoder_tokens = input_data_objects[1][0], input_data_objects[1][1]
    n_encoder_tokens = input_data_objects[1][2]

    seq2seq_model = encoder_decoder(n_encoder_tokens, n_decoder_tokens)
    seq2seq_model.fit([x_encoder, x_decoder], y_decoder, epochs=epochs, shuffle=True)
    

if __name__ == '__main__':
    
    train_encoder_decoder()
    
  