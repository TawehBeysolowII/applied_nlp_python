#Text Generation Demo
#Taweh Beysolow II 

#Import the necessary modules 
import numpy as np, pandas as pan, string
from chapter4.word_embeddings import load_data
from nltk.tokenize import word_tokenize

max_pages = 10
pdf_file = 'harry_potter.pdf'
punctuation = set(string.punctuation)
misc = '''... '' -- '''.split()

def preprocess_data(pdf_file=pdf_file, max_pages=max_pages):
    preprocessed_tokens = [] 
    text_data = load_data(pdf_file=pdf_file, max_pages=max_pages)
    for word in word_tokenize(text_data):
        if word not in punctuation and word not in misc: preprocessed_tokens.append(word)
    preprocessed_tokens = ' '.join([word for word in preprocessed_tokens])
    print(preprocessed_tokens)
    print(len(preprocessed_tokens))
    
if __name__ == '__main__':
    
    print(preprocess_data())