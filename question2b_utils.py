import pickle

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

SUBMIT_FILE_NAME = "q2b_submit.csv"
_DUMP_FILE = "datasets/duplicate.pickle.dat"
TRAIN_FILE_PATH = 'datasets/quora_features.csv'
TEST_FILE_PATH = 'datasets/quora_features_test.csv'


def dump_predictions_to_csv(data):
    print("Writting data to {0}".format(SUBMIT_FILE_NAME))
    data_to_submit = pd.DataFrame(data)
    data_to_submit.to_csv(SUBMIT_FILE_NAME, index=False)


def load_model():
    return pickle.load(open(_DUMP_FILE, "rb"))


def store_model(bst):
    return pickle.dump(bst, open(_DUMP_FILE, "wb"))


def get_target_values(data):
    return data.IsDuplicate.values


def get_data_for_model(data):
    return data[
        [
         'len_q1',
         'len_q2',
         'diff_len',
         'len_char_q1',
         'len_char_q2',
         'len_word_q1',
         'len_word_q2',
         'common_words',
         'fuzz_qratio',
         'fuzz_WRatio',
         'fuzz_partial_ratio',
         'fuzz_partial_token_set_ratio',
         'fuzz_partial_token_sort_ratio',
         'fuzz_token_set_ratio',
         'fuzz_token_sort_ratio',
         'wmd',
         'cosine_distance',
         'cityblock_distance',
         'jaccard_distance',
         'canberra_distance',
         'euclidean_distance',
         'minkowski_distance',
         'braycurtis_distance',
         'skew_q1vec',
         'skew_q2vec',
         'kur_q1vec',
         'kur_q2vec'
         ]
    ]
