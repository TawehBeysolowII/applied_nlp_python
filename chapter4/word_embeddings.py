#Word Embeddings 
#Taweh Beysolow II 

#Import the necessary modules 
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf, numpy as np

#Parameters 
epochs = 100
batch_size = 32
window_size = 2
learning_rate = 1e-4
embedding_dim = 4

def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])

def load_data(max_pages=100):
    return_string = StringIO()
    device = TextConverter(PDFResourceManager(), return_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(PDFResourceManager(), device=device)
    filepath = file('/Users/tawehbeysolow/Desktop/applied_nlp_python/datasets/economics_textbook.pdf', 'rb')
    for page in PDFPage.get_pages(filepath, set(), maxpages=max_pages, caching=True, check_extractable=True):
        interpreter.process_page(page)
    text_data = return_string.getvalue()
    filepath.close(), device.close(), return_string.close()
    return remove_non_ascii(text_data)
    
def gensim_preprocess_data():
    data = load_data()
    sentences = [sent_tokenize(remove_non_ascii(data))][0]
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
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
    data = load_data()
    sentences = [sent_tokenize(remove_non_ascii(data))][0]
    tokenized_sentences = list([word_tokenize(sentence) for sentence in sentences])
    output_data = []
    
    #Creating word pairs for word2vec model
    for sentence in tokenized_sentences:
        for index, word in enumerate(sentence):
            for _word in sentence[max(index - window_size, 0),
                                  min(index + window_size, len(sentence)) + 1]:
                if _word != word:
                    output_data.append([word, _word])

    
    return output_data

def tensorflow_word_embedding(learning_rate=learning_rate, embedding_dim=embedding_dim):
    data = tf_preprocess_data()
    vocab_size = len(np.unique(data))

    #Defining tensorflow variables and placeholder
    X = tf.placeholder(tf.int32, shape=[None, vocab_size])
    Y = tf.placeholder(tf.float32, shape=[None, vocab_size])
    
    weights = {'hidden': tf.Variable(tf.random_normal([vocab_size, embedding_dim])),
               'output': tf.Variable(tf.random_normal([embedding_dim, vocab_size]))}

    biases = {'hidden': tf.Variable(tf.random_normal([embedding_dim])),
              'output': tf.Variable(tf.random_normal([vocab_size]))}
              
    input_layer = tf.add(tf.matmul(X, weights['hidden']), biases['hidden'])
    output_layer = tf.add(tf.matmul(input_layer, weights['output']), biases['hidden'])
    
    #Defining error, optimizer, and other objects to be used during training 
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1)), tf.float32))
    cross_entropy = tf.reduce_mean(tf.cast(tf.nn.sampled_softmax_loss(labels=Y, inputs=output_layer, num_classes=vocab_size), tf.float32))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    
    #Executing graph 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initilizer())
        x, y = data[0], data[1]

        for epoch in range(epochs):          
            rows = np.random.randint(0, len(x)-1, len(x)-1)
            _train_x, _train_y = x[rows], y[rows]

            #Batch training 
            for start, end in zip(range(0, len(_train_x), batch_size), range(batch_size, len(_train_x), batch_size)):
                
                _accuracy, _cross_entropy, _optimizer = sess.run([accuracy, cross_entropy, optimizer], 
                                                         feed_dict={X:_train_x[start:end], Y: _train_y[start:end]})
                
            if epoch%10 == 0 and epoch > 1:
                print('Epoch: ' + str(epoch) + 
                        '\nError: ' + str(_cross_entropy) +
                        '\nAccuracy: ' + str(_accuracy) + '\n')
        
    
    
          
if __name__ == '__main__':
    
    #gensim_word_embedding()
    tensorflow_word_embedding()