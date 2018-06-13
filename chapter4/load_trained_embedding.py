#Example of loading a trained word embedding 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, collections
from chapter4.word_embeddings import load_data, cosine_similarity
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from tensorflow.contrib import rnn
from scipy import spatial

#Parameters
learning_rate = 1e-4; n_input = 5; 
n_hidden = 300; epochs = 100 
offset=10


sample_text = '''Living in different places has been the greatest experience 
that I have had in my life. It has allowed me to understand people from 
different walks of life, as well as to question some of my own biases I have had 
with respect to people who did not grow up as I did. If possible, everyone should 
take an opportunity to travel somewhere separate from where they grew up.'''.replace('\n', '')

def load_embedding(embedding_path='/Users/tawehbeysolow/Downloads/glove.6B.50D.txt'):
    vocabulary, embedding = [], []
    for line in open(embedding_path, 'rb').readlines():
        row = line.strip().split(' ')
        vocabulary.append(row[0]), embedding.append(row[1:])
    vocabulary_length, embedding_dim = len(vocabulary), len(embedding[0])
    return vocabulary, np.asarray(embedding, dtype=float), vocabulary_length, embedding_dim

def visualize_embedding_example():
    vocabulary, embedding, vocabulary_length, embedding_dim = load_embedding()

    #Showing example of pretrained word embedding vectors
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(embedding)
    plt.scatter(pca_embedding[0:50, 0], pca_embedding[0:50, 1])
    for i, word in enumerate(vocabulary[0:50]):
        plt.annotate(word, xy=(pca_embedding[i, 0], pca_embedding[i, 1]))
        
    #Comparing cosine similarity 
    for k in range(100, 150):
        text = str('Cosine Similarity Between %s and %s: %s')%(vocabulary[k],
                                                            vocabulary[k-1], 
                                                cosine_similarity(embedding[k], 
                                                                  embedding[k-1]))
        print(text)

        
def training_data_example(sample_text=sample_text, 
                          learning_rate=learning_rate, 
                          n_input=n_input, 
                          n_hidden=n_hidden, 
                          epochs=epochs,
                          offset=offset):
    
    vocabulary, embedding, vocabulary_length, embedding_dim = load_embedding()
    _sample_text = np.array(sample_text.split()).reshape(sample_text, [-1, ])
    _embeddings, embeddings_tmp = [], []

    def sample_text_dictionary(data=remove_stop_words(_sample_text)):
        count, dictionary = collections.Counter(data).most_common(), {} #creates list of word/count pairs;
        for word, _ in count:
            dictionary[word] = len(dictionary) #len(dictionary) increases each iteration
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        tf_dictionary_list = sorted(dictionary.items(), key = lambda x : x[1])
        vocabulary_length = len(dictionary)
        return dictionary, reverse_dictionary, vocabulary_length, tf_dictionary_list
        
    dictionary, reverse_dictionary, _vocabulary_length, dictionary_list = sample_text_dictionary()
    
    for i in range(_vocabulary_length):
        word = dictionary_list[i][0]
        if word in vocabulary:
            _embeddings.append(embedding_dict[item])
        else:
            embeddings_tmp.append(np.random.uniform(low=-0.2, high=0.2,size=embedding_dim))
     
    embedding_array = np.asarray(embeddings_tmp)     
    decision_tree = spatial.KDTree(embedding)

    #Creating recurrent neural network for task
    X = tf.placeholder(tf.float32, shape=(None, None, n_input))
    Y = tf.placeholder(tf.float32, shape=(None, embedding_dim))
    weights = {'output': tf.Variable(tf.random_normal(n_hidden, embedding_dim))}
    biases = {'output': tf.Variable(tf.random_normal(embedding_dim))}
              
    with tf.name_scope("embedding"):
        _weights = tf.Variable(tf.constant(0.0, shape=[vocabulary_length, embedding_dim]), trainable=True, name='_weights')
        _embedding = tf.placeholder(tf.float32, [vocabulary_length, embedding_dim])
        embedding_initializer = _weights.assign(_embedding)
        embedding_characters = tf.nn.embedding_looking(_weights, X)
        _sample_text = np.array(sample_text.split()).reshape(sample_text, [-1, ])
        
    # reshape input data
    x_unstacked = tf.unstack(embedding_characters, n_input, 1)
    rnn_cell =  rnn.BasicLSTMCell(num_units=n_hidden, state_is_tuple=True, reuse=None)
    outputs, states = rnn.static_rnn(rnn_cell, x_unstacked, dtype=tf.float32)
    output_layer = tf.matmul(outputs[-1], weights['out']) + biases['out'] 
     
    # Create loss function and optimizer
    error = tf.reduce_mean(tf.pow(output_layer-Y, 2))//len(vocabulary)
    adam_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initiliazer())
        
        for epoch in range(epochs):
           
            sess.run(embedding_initializer, feed_dict={_embedding: embedding})
            
             #Creatin input and output training data
            x_train = [[dictionary[str(vocabulary[i])]] for i in range(offset, offset+n_input)]
            x_train = np.reshape(np.array(x_train), [-1, n_input])
            y_train = offset+n_input
            y_train = dictionary[training_data[y_train]]
            y_train = embedding[y_train,:]
            y_train = np.reshape(y_train,[1,-1])
    
            
            _,loss, pred_ = sess.run([adam_optimizer, error, output_layer], 
                                     feed_dict = {X: x_train, Y: y_train})
            
            
             
            if epoch%10 == 0 and epoch > 0:
                words_in = [str(x_train[i]) for i in range(offset, offset+n_input)] 
                #target_word = str(x_train[y_position])
                nearest_dist,nearest_idx = decision_tree.query(pred_[0],3)
                nearest_words = [reverse_dictionary[idx] for idx in nearest_idx]
                  
                #print("%s - [%s] vs [%s]" % (words_in, target_word, nearest_words))
                #print("Average Loss= " + "{:.6f}".format(loss_total/display_step))
          
                offset += (n_input+1) 
      



if __name__ == '__main__':

    #visualize_embedding_example()
    
    training_data_example()
