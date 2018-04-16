
import os, numpy as np

def remove_non_ascii(text):
    return ''.join([word for word in text if ord(word) < 128])
    
def load_data():
    negative_review_strings = os.listdir('/applied_nlp_python/datasets/review_data/tokens/neg')
    positive_review_strings = os.listdir('/applied_nlp_python/datasets/review_data/tokens/pos')
    negative_reviews, positive_reviews = [], []
    
    for positive_review in positive_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/pos/'+str(positive_review), 'r') as positive_file:
            positive_reviews.append(remove_non_ascii(positive_file.read()))
    
    for negative_review in negative_review_strings:
        with open('/Users/tawehbeysolow/Downloads/review_data/tokens/neg/'+str(negative_review), 'r') as negative_file:
            negative_reviews.append(remove_non_ascii(negative_file.read()))
    
    negative_labels, positive_labels = np.repeat(0, len(negative_reviews)), np.repeat(1, len(positive_reviews))
    labels = np.concatenate([negative_labels, positive_labels])
    reviews = np.concatenate([negative_reviews, positive_reviews])
    rows = np.random.random_integers(0, len(reviews)-1, len(reviews)-1)
    return reviews[rows], labels[rows]