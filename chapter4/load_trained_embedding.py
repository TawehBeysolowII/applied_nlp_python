#Example of loading a trained word embedding 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from chapter4.word_embeddings import load_data, cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from sklearn.decomposition import PCA

#Parameters
learning_rate = 1e-4
n_input = 5
n_hidden = 300
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

        
def training_data_example(sample_text=sample_text, learning_rate=learning_rate, n_input=n_input, n_hidden=n_hidden):
    vocabulary, embedding, vocabulary_length, embedding_dim = load_embedding()
    weights = tf.Variable(tf.constant(0.0, shape=[vocabulary_length, embedding_dim]), trainable=False, name='weights')
    _embedding = tf.placeholder(tf.float32, [vocabulary_length, embedding_dim])
    embedding_initialiazer = weights.assign(_embedding)
    _sample_text = np.array(sample_text.split()).reshape(sample_text, [-1, ])
    
    def sample_text_dictionary(data=_sample_text):
        count, dictionary = collections.Counter(words).most_common(), {} #creates list of word/count pairs;
        for word, _ in count:
            dictionary[word] = len(dictionary) #len(dictionary) increases each iteration
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, reverse_dictionary


    #Creating recurrent neural network for task
        

    '''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initiliazer())
        
        sess.run(embedding_initializer, feed_dict={_embedding: embedding})
    '''


if __name__ == '__main__':

    visualize_embedding_example()
