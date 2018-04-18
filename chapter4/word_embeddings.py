#Word Embeddings 
#Taweh Beysolow II 

#Import the necessary modules 
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np, string

#Parameters 
np.random.seed(2018)
epochs = 100
batch_size = 32
window_size = 2
learning_rate = 1e-4
embedding_dim = 300
stop_words = stopwords.words('english')
punctuation = set(string.punctuation)
page_len = 10

def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])

def load_data(max_pages=page_len):
    return_string = StringIO()
    device = TextConverter(PDFResourceManager(), return_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath = file('/Users/tawehbeysolow/Desktop/applied_nlp_python/datasets/economics_textbook.pdf', 'rb')
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text_data = return_string.getvalue()
    filepath.close(), device.close(), return_string.close()
    text_data = ' '.join([word for word in word_tokenize(remove_non_ascii(text_data)) if word not in stop_words])
    return text_data
    
def gensim_preprocess_data():
    data = load_data()
    sentences = [sent_tokenize(remove_non_ascii(data))][0]
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences if word_tokenize(sentence) not in punctuation])
    return tokenized_sentences
    
def gensim_word_embedding():
    sentences = gensim_preprocess_data()[1002:1003]
    skip_gram = Word2Vec(sentences=sentences, window=window_size, min_count=1)
    word_embedding = skip_gram[skip_gram.wv.vocab]
    pca = PCA(n_components=2)
    word_embedding = pca.fit_transform(word_embedding)
    
    #Plotting results from trained word embedding
    plt.scatter(word_embedding[:, 0], word_embedding[:, 1])
    word_list = list(skip_gram.wv.vocab)
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(word_embedding[i, 0], word_embedding[i, 1]))
        
def tf_preprocess_data(window_size=window_size):
        
    def one_hot_encoder(index, vocab_size):
        vector = np.zeros(vocab_size)
        vector[index] = 1
        return vector
        
    text_data = load_data()
    vocab_size = len(word_tokenize(text_data))
    word_dictionary, int_dictionary = {}, {}
    for index, word in enumerate(word_tokenize(text_data)):
        word_dictionary[word], int_dictionary[index] = index, word
           
    sentences = sent_tokenize(text_data)
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    n_gram_data = []
    
    #Creating word pairs for word2vec model
    for sentence in tokenized_sentences:
        for index, word in enumerate(sentence):
            if word not in punctuation: 
                for _word in sentence[max(index - window_size, 0):
                                      min(index + window_size, len(sentence)) + 1]:
                    if _word != word:
                        n_gram_data.append([word, _word])

    #One-hot encoding data and creating dataset intrepretable by skip-gram model
    x, y = np.zeros([len(n_gram_data), vocab_size]), np.zeros([len(n_gram_data), vocab_size])
    
    for i in range(0, len(n_gram_data)):
        x[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][0]], vocab_size=vocab_size)      
        y[i, :] = one_hot_encoder(word_dictionary[n_gram_data[i][1]], vocab_size=vocab_size)

    return x, y, vocab_size, word_dictionary

def tensorflow_word_embedding(learning_rate=learning_rate, embedding_dim=embedding_dim):
    x, y, vocab_size, word_dictionary = tf_preprocess_data()
    
    #Defining tensorflow variables and placeholder
    X = tf.placeholder(tf.float32, shape=(None, vocab_size))
    Y = tf.placeholder(tf.float32, shape=(None, vocab_size))
    
    weights = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])),
               'output': tf.Variable(tf.random_normal([embedding_dim, vocab_size]))}

    biases = {'hidden': tf.Variable(tf.random_normal([embedding_dim])),
              'output': tf.Variable(tf.random_normal([vocab_size]))}
              
    input_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
    output_layer = tf.add(tf.matmul(input_layer, weights['output']), biases['output'])
    
    #Defining error, optimizer, and other objects to be used during training 
    cross_entropy = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=Y), tf.float32))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    
    #Executing graph 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):          
            rows = np.random.randint(0, len(x)-50, len(x)-50)
            _train_x, _train_y = x[rows], y[rows]


            #Batch training
            for start, end in zip(range(0, len(_train_x), batch_size), 
                                  range(batch_size, len(_train_x), batch_size)):
                
                _cross_entropy, _optimizer = sess.run([cross_entropy, optimizer], 
                                                      feed_dict={X:_train_x[start:end], Y: _train_y[start:end]})
                
            if epoch%10==0 and epoch > 1:
                print('Epoch: ' + str(epoch) + 
                        '\nError: ' + str(_cross_entropy) + '\n')
        

        word_embedding = sess.run(tf.add(weights['input'], biases['input']))
        pca = PCA(n_components=2)
        word_embedding = pca.fit_transform(word_embedding)
        
        #Plotting results from trained word embedding
        plt.scatter(word_embedding[:, 0], word_embedding[:, 1])
        word_list = word_dictionary.keys()
        for i, word in enumerate(word_list):
            plt.annotate(word, xy=(word_embedding[i, 0], word_embedding[i, 1]))

if __name__ == '__main__':
    
    #gensim_word_embedding()
    tensorflow_word_embedding()