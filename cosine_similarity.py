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


def get_vector_from_brand(df:pd.DataFrame, name:str):
    # look for the column embeddings in the row "name" from the column "Marque" in the dataframe "df" and return it, if patent exists
    if type(name) != str:
        # print("Patent name must be a string. Here is the type: {} and the content: {}".format(type(name), name))
        return None
    try:
        ret = df.loc[df["Marque"] == name.lower(), "Embeddings"].values[0]
    except IndexError:
        print("Patent not found")
        ret = None


def cosine_similarity(brand_a:str, brand_b:str, df:pd.DataFrame=None):
    if df is None:
        df = get_embeddings_dataframe()
    vector_a = get_vector_from_brand(df, brand_a)
    vector_b = get_vector_from_brand(df, brand_b)
    # Check if patent exists
    if vector_a is None or vector_b is None:
        return None
    # compute the cosine similarity between two vectors
    numerator = np.dot(vector_a, vector_b)
    den_a, den_b = 0, 0
    for i in range(len(vector_a)):
        den_a += vector_a[i] ** 2
        den_b += vector_b[i] ** 2
    denominator = sqrt((den_a ** 0.5) * (den_b ** 0.5))
    return numerator / denominator


# df = get_embeddings_dataframe()

patent_a = "reenix"
patent_b = "monjour"

# vector_a = get_vector_from_patent(df, "kaNyar")
# vector_b = get_vector_from_patent(df, "kanYar")


print("Cosine similarity: ", cosine_similarity(patent_a, patent_b))