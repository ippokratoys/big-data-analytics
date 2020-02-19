from sklearn.metrics.pairwise import cosine_similarity

import timing as timing
from question_2_a_utils import read_data_and_create_vectorizer


# couldn't find something ready, so instead I did a quick hack.
def jacardi_similarity(x, y):
    return cosine_similarity(1 * (x > 0), 1 * (y > 0))


def num_of_similars(existing_vectors, new_vector, similarity_fun):
    num_of_similars = 0
    for existing in existing_vectors:
        similarity = similarity_fun(existing, new_vector)
        if (similarity >= 0.8):
            num_of_similars += 1
    return num_of_similars


def num_of_similars_jac(existing_vectors, new_vector):
    num_of_similars = 0
    # Remove if your computer can handle it
    # for existing in existing_vectors:
    for existing in existing_vectors[:1]:
        existing_normalized = 1 * (existing > 0)
        new_normalized = 1 * (new_vector > 0)
        similarity = cosine_similarity(existing_normalized, new_normalized)
        if (similarity >= 0.8):
            num_of_similars += 1
    return num_of_similars


train_vectors, test_data, vectorizer = read_data_and_create_vectorizer()

timing.log_start("Start comparing...")
count = 0
for new_vector in vectorizer.transform(test_data['Content']):
    # Switch beetwen the following lines to run different similarities algorithms
    res = num_of_similars(train_vectors, new_vector, cosine_similarity)
    # res = num_of_similars(train_vectors, new_vector, jacardi_similarity)
    if (res > 0):
        count += 1

timing.log_finish()
print("Found count {} similar".format(count))
