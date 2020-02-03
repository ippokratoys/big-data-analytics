from datasketch import MinHash, MinHashLSH
from scipy.sparse import lil_matrix

import timing as timing
from question_2_a_utils import read_data_and_create_vectorizer


def create_hash_from_vector(vector_to_be_hashed: lil_matrix, num_of_perm):
    hashed_vector: MinHash = MinHash(num_perm=num_of_perm)
    for vector_element in vector_to_be_hashed.todense():
        hashed_vector.update(vector_element)
    return hashed_vector


def run_lsh(num_of_perm, train_vectors, test_data, vectorizer):
    print("----------------{}-------------------".format(num_of_perm))
    timing.log_start("Building lsh...")
    lsh = MinHashLSH(threshold=0, num_perm=num_of_perm)
    for idx, vector in enumerate(train_vectors):
        hashed_vector: MinHash = create_hash_from_vector(vector, num_of_perm)
        lsh.insert(str(idx), hashed_vector)
    time_to_build_index = timing.log_finish()

    timing.log_start("Start comparing...")
    count = 0
    for new_vector in vectorizer.transform(test_data['Content']):
        hashed_vector = create_hash_from_vector(new_vector, num_of_perm)
        res = lsh.query(hashed_vector)
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

# run for different valus of permutation numbers
metrics = [run_lsh(i, train_vectors, test_data, vectorizer) for i in [16, 32, 64]]
print(metrics)
