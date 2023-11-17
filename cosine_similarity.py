import pandas as pd
import numpy as np
from math import sqrt

def get_embeddings_dataframe():
    # return the embeddings dataframe. If it doesn't exist, create it
    try:
        print("Reading embeddings.csv\n")
        df_result = pd.read_csv("embeddings.csv", sep=';', encoding='UTF-8')
        # the column "Embeddings" is a string. Convert it to a list of floats
        df_result["Embeddings"] = df_result["Embeddings"].apply(lambda x: [float(i) for i in x[1:-1].split(", ")])
    except FileNotFoundError:
        print("embeddings.csv not found. Computing embeddings.\n")
        from read_data import read_csv, preprocess_csv, preprocess_text
        from model import get_embeddings
        df = read_csv()
        df = preprocess_csv(df)
        df = preprocess_text(df)
        df_result = get_embeddings(df)
    return df_result


def vector_from_patent(patent):
    # return the vector from the patent
    vector = 0
    n = len(patent)
    for i in range(n):
        vector += patent[i]
    return vector / n


def cosine_similarity(v1, v2):
    # compute the cosine similarity between two vectors
    numerator = np.dot(v1, v2)
    den_a, den_b = 0, 0
    for i in range(len(v1)):
        den_a += v1[i] ** 2
        den_b += v2[i] ** 2
    denominator = sqrt((den_a ** 0.5) * (den_b ** 0.5))
    return numerator / denominator


def get_vector_from_patent(df, name):
    # return only the embeddings, not the name neither the index
    return df.loc[df['Marque'] == name, 'Embeddings'].iloc[0]

df = get_embeddings_dataframe()

vector_a = get_vector_from_patent(df, "KANYAR")
vector_b = get_vector_from_patent(df, "KANYAR")


print("Cosine similarity: ", cosine_similarity(vector_a, vector_b))

# export the whole column Marque from the dataframe and save it into a file
df["Marque"].to_csv("marques.csv", sep=';', encoding='UTF-8', index=False)