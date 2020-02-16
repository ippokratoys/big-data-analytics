import pickle

import gensim
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

######### run for all datasets ############
FILE_TO_READ = 'datasets/q2b/train.csv'
FILE_TO_WRITE_Q1 = 'datasets/q1_w2v.pkl'
FILE_TO_WRITE_Q2 = 'datasets/q2_w2v.pkl'


######### run for all datasets ############
# FILE_TO_READ = 'datasets/q2b/test_without_labels.csv'
# FILE_TO_WRITE_Q1 = 'datasets/q1_w2v_test.pkl'
# FILE_TO_WRITE_Q2 = 'datasets/q2_w2v_test.pkl'


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


stop_words = stopwords.words('english')
nltk.download('punkt')

data = pd.read_csv(FILE_TO_READ)

print("Step 1 (load google new vector)")
model = gensim.models.KeyedVectors.load_word2vec_format('datasets/GoogleNews-vectors-negative300.bin', binary=True)

print("Step 2 (intitialize vectors for q1)")
question1_vectors = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.Question1.values)):
    question1_vectors[i, :] = sent2vec(q)

print("Step 3 (intitialize vectors for q2)")
question2_vectors = np.zeros((data.shape[0], 300))
for i, q in tqdm(enumerate(data.Question2.values)):
    question2_vectors[i, :] = sent2vec(q)

print("dumping")
pickle.dump(question1_vectors, open(FILE_TO_WRITE_Q1, 'wb'), -1)
pickle.dump(question2_vectors, open(FILE_TO_WRITE_Q2, 'wb'), -1)
