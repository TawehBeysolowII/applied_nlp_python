#Doc2Vec Example 
#Taweh Beysolow II 

#Import the necessary modules 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from gensim.models import Doc2Vec
from collections import namedtuple 
from chapter4.word_embeddings import load_data, cosine_similarity
import time

#Parameters
stop_words = stopwords.words('english')
learning_rate = 1e-4
epochs = 200
max_pages = 20

def gensim_preprocess_data(max_pages):
    sentences = namedtuple('sentence', 'words tags')
    _sentences = sent_tokenize(load_data(max_pages=max_pages))
    documents = []
    for i, text in enumerate(_sentences):
        words, tags = text.lower().split(), [i]
        documents.append(sentences(words, tags))
    return documents

def train_model(max_pages=max_pages, epochs=epochs, learning_rate=learning_rate):
    sentences = gensim_preprocess_data(max_pages=10)[50:70] 
    model = Doc2Vec(alpha=learning_rate, min_alpha=learning_rate/float(3))
    model.build_vocab(sentences)
    model.train(documents=sentences, total_examples=len(sentences), epochs=epochs)
        
    #Showing distance between different documents 
    for i in range(1, len(sentences)-1):
        print(str('Document ' + str(sentences[i-1]) + '\n'))
        print(str('Document ' + str(sentences[i]) + '\n'))
        print('Cosine Similarity Between Documents: ' + 
              '\n' + str(cosine_similarity(model.docvecs[i-1], model.docvecs[i])))
        
        time.sleep(10)
    
if __name__ == '__main__': 
    
    train_model()