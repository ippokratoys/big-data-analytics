from nearpy import Engine
from nearpy.distances import CosineDistance
from nearpy.filters import DistanceThresholdFilter
from nearpy.hashes import RandomBinaryProjections

import timing as timing
from question_2_a_utils import read_data_and_create_vectorizer


def run_lsh(num_of_perm, train_vectors, test_data, vectorizer):
    print("----------------{}-------------------".format(num_of_perm))
    timing.log_start("Building lsh...")
    dim = train_vectors.shape[1]
    engine = Engine(
        dim,
        lshashes=[RandomBinaryProjections(str(i), i, rand_seed=4242)],
        distance=CosineDistance(),
        vector_filters=[DistanceThresholdFilter(0.8)]
    )
    for idx, vector in enumerate(train_vectors):
        engine.store_vector(vector.reshape(dim, 1), str(idx))

    time_to_build_index = timing.log_finish()

    timing.log_start("Start comparing...")
    count = 0
    for new_vector in vectorizer.transform(test_data['Content']):
        res = engine.neighbours(new_vector.reshape(dim, 1))
        if (len(res) > 0):
            count += 1
    time_to_run_queries = timing.log_finish()
    print("Found {} similar".format(count))

    avg_query_time = time_to_run_queries / len(test_data)
    return {
        "permutations": num_of_perm,
        "time_to_build": time_to_build_index,
        "time_to_run_queries": time_to_run_queries,
        "avg_query_time": avg_query_time,
    }


train_vectors, test_data, vectorizer = read_data_and_create_vectorizer()

# run for different values of K
metrics = []
for i in [10, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    metrics.append(run_lsh(i, train_vectors, test_data, vectorizer))

print(metrics)
