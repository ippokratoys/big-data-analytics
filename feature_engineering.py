"""
Detecting duplicate quora questions
feature engineering
@author: Abhishek Thakur

The source code/ github repository can be found here
See Also : https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question
"""
import pickle

import gensim
import nltk
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from nltk import word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from scipy.stats import skew, kurtosis

# DATASET = 'datasets/q2b/train.csv'
# VECTOR_Q1 = 'datasets/q1_w2v.pkl'
# VECTOR_Q2 = 'datasets/q2_w2v.pkl'
# QUORA_FEATURES_CSV = 'datasets/quora_features.csv'
#
DATASET = 'datasets/q2b/test_without_labels.csv'
VECTOR_Q1 = 'datasets/q1_w2v_test.pkl'
VECTOR_Q2 = 'datasets/q2_w2v_test.pkl'
QUORA_FEATURES_CSV = 'datasets/quora_features_test.csv'

stop_words = stopwords.words('english')
nltk.download('punkt')


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


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


data = pd.read_csv(DATASET)

print("Step 1")
data['len_q1'] = data.Question1.apply(lambda x: len(str(x)))

print("Step 2")
data['len_q2'] = data.Question2.apply(lambda x: len(str(x)))

print("Step 3")
data['diff_len'] = data.len_q1 - data.len_q2

print("Step 4")
data['len_char_q1'] = data.Question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))

print("Step 5")
data['len_char_q2'] = data.Question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))

print("Step 6")
data['len_word_q1'] = data.Question1.apply(lambda x: len(str(x).split()))

print("Step 7")
data['len_word_q2'] = data.Question2.apply(lambda x: len(str(x).split()))

print("Step 8")
data['common_words'] = data.apply(
    lambda x: len(set(str(x['Question1']).lower().split()).intersection(set(str(x['Question2']).lower().split()))),
    axis=1)

print("Step 9")
data['fuzz_qratio'] = data.apply(lambda x: fuzz.QRatio(str(x['Question1']), str(x['Question2'])), axis=1)

print("Step 10")
data['fuzz_WRatio'] = data.apply(lambda x: fuzz.WRatio(str(x['Question1']), str(x['Question2'])), axis=1)

print("Step 11")
data['fuzz_partial_ratio'] = data.apply(lambda x: fuzz.partial_ratio(str(x['Question1']), str(x['Question2'])), axis=1)

print("Step 12")
data['fuzz_partial_token_set_ratio'] = data.apply(
    lambda x: fuzz.partial_token_set_ratio(str(x['Question1']), str(x['Question2'])), axis=1)

print("Step 13")
data['fuzz_partial_token_sort_ratio'] = data.apply(
    lambda x: fuzz.partial_token_sort_ratio(str(x['Question1']), str(x['Question2'])), axis=1)

print("Step 14")
data['fuzz_token_set_ratio'] = data.apply(lambda x: fuzz.token_set_ratio(str(x['Question1']), str(x['Question2'])),
                                          axis=1)

print("Step 15")
data['fuzz_token_sort_ratio'] = data.apply(lambda x: fuzz.token_sort_ratio(str(x['Question1']), str(x['Question2'])),
                                           axis=1)

print("Step 16")
model = gensim.models.KeyedVectors.load_word2vec_format('datasets/GoogleNews-vectors-negative300.bin', binary=True)

print("Step 16a")
data['wmd'] = data.apply(lambda x: wmd(x['Question1'], x['Question2']), axis=1)

print("Step 18")
question1_vectors = pickle.load(open(VECTOR_Q1, 'rb'))
question2_vectors = pickle.load(open(VECTOR_Q2, 'rb'))

print("Step 21")
data['cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                          np.nan_to_num(question2_vectors))]

print("Step 22")
data['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

print("Step 23")
data['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                            np.nan_to_num(question2_vectors))]

print("Step 24")
data['canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]

print("Step 25")
data['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                np.nan_to_num(question2_vectors))]

print("Step 26")
data['minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                   np.nan_to_num(question2_vectors))]

print("Step 27")
data['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                                  np.nan_to_num(question2_vectors))]

print("Step 28")
data['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]

print("Step 29")
data['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]

print("Step 30")
data['kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]

print("Step 31")
data['kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]

print("Dumping to file...")
data.to_csv(QUORA_FEATURES_CSV, index=False)
