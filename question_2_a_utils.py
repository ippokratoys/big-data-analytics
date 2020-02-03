import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import timing as timing

# Unfortunately my laptop can't handle all this data.
PERCENTAGE_OF_DATA = 10


def read_data_and_create_vectorizer():
    train_data = pd.read_csv('datasets/q2a/corpusTrain.csv')
    data_to_use = int(len(train_data) * PERCENTAGE_OF_DATA / 100)
    train_data = train_data[:data_to_use]

    test_data = pd.read_csv('datasets/q2a/corpusTest.csv')

    print("Running for {} train data({}%)".format(
        len(train_data),
        PERCENTAGE_OF_DATA
    ))
    print("Running for {} test data".format(len(test_data)))

    vectorizer = TfidfVectorizer()
    timing.log_start("Fitting vectorizer")
    train_vectors = vectorizer.fit_transform(train_data['Content'])
    timing.log_finish()

    return train_vectors, test_data, vectorizer
