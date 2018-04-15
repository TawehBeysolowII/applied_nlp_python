#Chapter 3 text preprocessing examples 
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np 

sample_text = '''I am a student from the University of Alabama. I 
was born in Ontario, Canada and I am a huge fan of the United States. 
I am going to get a degree in Philosophy to improve my chances of 
becoming a Philosophy professor. I have been working towards this goal
for 4 years. I am currently enrolled in a PhD program. It is very difficult, 
but I am confident that it will be a good decision'''

from nltk.tokenize import word_tokenize, sent_tokenize

sample_word_tokens = word_tokenize(sample_text)
print(sample_word_tokens)
sample_sent_tokens = sent_tokenize(sample_text)
print(sample_sent_tokens)

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

def mistake():
    stop_words = stopwords.words('english')
    word_tokens = [word for word in sample_word_tokens if word not in stop_words]
    print(word_tokens)
    return word_tokens
    
mistake = mistake()
len(mistake)

def advised_preprocessing(sample_word_tokens=sample_word_tokens):
    stop_words = [word.upper() for word in stopwords.words('english')]
    word_tokens = [word for word in sample_word_tokens if word.upper() not in stop_words]
    print(word_tokens)
    return word_tokens

sample_word_tokens = advised_preprocessing()

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
sample_word_tokens = tokenizer.tokenize(str(sample_word_tokens))
sample_word_tokens = [word.lower() for word in sample_word_tokens]
print(sample_word_tokens)

import collections, re

def bag_of_words(text):
    _bag_of_words = [collections.Counter(re.findall(r'\w+', word)) for word in text]
    bag_of_words = sum(_bag_of_words, collections.Counter())
    return bag_of_words
    
sample_word_tokens_bow = bag_of_words(text=sample_word_tokens)
print(sample_word_tokens_bow)

sample_text_bow = bag_of_words(text=word_tokenize(sample_text))

from sklearn.feature_extraction.text import CountVectorizer

def bow_sklearn(text=sample_sent_tokens):
    c = CountVectorizer(stop_words='english', token_pattern=r'\w+')
    converted_data = c.fit_transform(text).todense()
    print(converted_data.shape)
    return converted_data, c.get_feature_names()

bow_data, feature_names = bow_sklearn()

'''
TF-IDF EXAMPLE
'''

text = '''I was a student at the University of Pennsylvania, but now work on 
Wall Street as a Lawyer. I have been living in New York for roughly five years
now, however I am looking forward to eventually retiring to Texas once I have 
saved up enough money to do so.'''


text2= '''I am a doctor who is considering retirement in the next couple of years. 
I went to the Yale University, however that was quite a long time ago. I have two children,
who both have three children each, making me a grandfather. I look forward to retiring 
and spending more time with them
'''

def preprocessing(sample_word_tokens=sample_word_tokens):
    stop_words = [word.upper() for word in stopwords.words('english')]
    word_tokens = [word for word in sample_word_tokens if word.upper() not in stop_words]
    return word_tokens

def preprocesser(text=sample_text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(str(text))
    text = [word.lower() for word in text]
    text = preprocessing(sample_word_tokens=text)
    text = [word.lower() for word in text]
    return text

document_list = list([sample_text, text])

def tf_idf(documents=document_list):
    #Finding unique terms across all documents
    terms = []
    preprocessed_documents = list([preprocesser(document) for document in documents])
    for document in preprocessed_documents:
        terms.append([term for term in np.unique(document)])
    
    terms = terms[0]+terms[1]
    terms = np.unique(terms)
    
    term_freq = np.zeros([1, len(terms)])
    for j in range(0, len(terms)):
        for k in range(0, len(document_list)):
            index = range(0, len(preprocessed_documents[k]))
            for i in index:
                if terms[j] == preprocessed_documents[k][i]:
                    term_freq[0][j] += 1

    term_count = dict([(terms[i], term_freq[0][i]) for i in range(0, len(terms))])
    print('Term Frequency: \n' + str(term_count))
    
    #Calculating inverse document frequency
    inv_df = [np.log(len(preprocessed_documents)/(term_count.values()[i])) 
    for i in range(0, len(term_count.values()))]
    
    inv_doc_freq = dict([(term_count.keys()[i], inv_df[i]) for i in range(0, len(term_count.keys()))])
    print('Inverse Document Frequency: \n' + str(inv_doc_freq))
    
    #Calculating Tf x Idf
    tf_idf = np.zeros([1, len(inv_doc_freq)])
    for i in range(0, len(tf_idf)):
        tf_idf[i] = inv_doc_freq.values()[i] * term_count[inv_doc_freq.keys()[i]]

    tf_idf_dict = dict([(inv_doc_freq.keys()[i], tf_idf[0][i]) for i in range(0, len(tf_idf[0]))])
    print('TFIDF Score: \n' + str(tf_idf_dict))

tf_idf()

from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf_sklearn(document=document_list):
    t = TfidfVectorizer(stop_words='english', token_pattern=r'\w+')
    x = t.fit_transform(document_list).todense()
    print(x)
    print(x.shape)
    
tf_idf_sklearn()
