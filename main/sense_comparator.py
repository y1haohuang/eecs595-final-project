import numpy as np
from gensim.models import KeyedVectors
from numpy.linalg import norm
from sklearn.cluster import KMeans
from tqdm import tqdm

def load_word_vectors(src_file_path, trg_file_path):
    src_emb = KeyedVectors.load_word2vec_format(src_file_path, binary=False)
    trg_emb = KeyedVectors.load_word2vec_format(trg_file_path, binary=False)

    return src_emb, trg_emb


def load_transform_matrix(transformation_matrix_path):
    return np.loadtxt(transformation_matrix_path)

def compare_sense_island(sense_word, src_emb, trg_emb, transformation_matrix,
                         c1, c2, island_size, island_num):
    num_island_1 = 0
    num_island_2 = 0
    islands = []
    # print("len(c1):", len(c1))
    for sentence in c1:
        # print("len(sentence):", len(sentence))
        if sense_word in sentence:
            index = sentence.index(sense_word)
            island = sentence[max(index - island_size, 0):index]
            island += sentence[index:min(index + island_size, len(sentence))]
            island_vec = np.array(
                [np.dot(src_emb[word], transformation_matrix) for word in
                 island if word in src_emb]).mean(axis=0)
            islands.append(island_vec)
            num_island_1 += 1

    for sentence in c2:
        if sense_word in sentence:
            index = sentence.index(sense_word)
            island = sentence[max(index - island_size, 0):index]
            island += sentence[index:min(index + island_size, len(sentence))]
            island_vec = np.array(
                [trg_emb[word] for word in island if word in trg_emb]).mean(
                axis=0)
            islands.append(island_vec)
            num_island_2 += 1

    islands = np.array(islands)

    kmeans = KMeans(n_clusters=island_num, random_state=0).fit(islands)
    island_label = kmeans.labels_.tolist()
    islands_profile = {}
    for i in range(island_num):
        islands_profile[i] = {"c1": 0, "c2": 0}
    for index, id in enumerate(island_label):
        if index < num_island_1:
            islands_profile[id]["c1"] += 1
        else:
            islands_profile[id]["c2"] += 1
    print()
    print("word:", sense_word)
    for i in range(island_num):
        print(i, "c1:", islands_profile[i]["c1"], "c2:", islands_profile[i]["c2"])
        if islands_profile[i]["c1"] + islands_profile[i]["c2"] < 4:
            continue
        if min(islands_profile[i]["c1"], islands_profile[i]["c2"]) < 1 and \
                max(islands_profile[i]["c1"], islands_profile[i]["c2"]) > 5:
            return 1

    return 0

def compare_sense(sense_word, src_emb, trg_emb, transformation_matrix, topn=10, use_op=False):
    similarity = 0.0

    try:
        src_vec = src_emb[sense_word]
        trg_vec = trg_emb[sense_word]
    except:  # here if key misssing
        print('Not present word:' + sense_word)
        print("Similarity:0.0000")
        print('--------------')
        return similarity, None, None

    transformed_vec = np.dot(src_vec, transformation_matrix)

    if use_op is False:
        # similarity after transformation between the target word in source and target space
        similarity = compute_cosine_sim(transformed_vec, trg_vec)
    else:
        similarity = compute_cosine_sim(src_vec, trg_vec)
        transformed_vec = src_vec

    # 10 most similar words to sense word in a target space
    most_similar_to_original_word = trg_emb.wv.most_similar(positive=[sense_word], negative=[], topn=topn)
    most_similar_to_original_word.append((sense_word, 1.0))

    # 10 most similar words to a transformed vector (from source space)
    most_similar_to_transformed_vector = trg_emb.wv.similar_by_vector(transformed_vec, topn=topn+1)

    return similarity, most_similar_to_original_word, most_similar_to_transformed_vector


def compute_cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def compute_euclidean_dis(a, b):
    return norm(a - b)



