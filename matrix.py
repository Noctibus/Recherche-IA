from cosine_similarity import cosine_similarity, get_embeddings_dataframe
import numpy as np
from tqdm import tqdm


def matrix():
    df = get_embeddings_dataframe()
    # make nupy matrix of size the amount of rows in df
    brands = df["Marque"].tolist()
    n = len(brands)
    mat = np.zeros((n, n))
    for i in tqdm(range(n)):
        for j in range(i, n):
            brand_a = brands[i]
            brand_b = brands[j]
            similarity = cosine_similarity(brand_a, brand_b, df=df)
            mat[i][j] = similarity
            mat[j][i] = similarity
    return matrix


def save_matrix_to_file(mat:np.ndarray):
    np.save("matrix", mat)


mat = matrix()
save_matrix_to_file(mat)