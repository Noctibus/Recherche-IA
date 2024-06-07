import pandas as pd
from read_data import read_csv, preprocess_csv, preprocess_text
from sentence_transformers import SentenceTransformer



def get_data():
    df = read_csv()
    df = preprocess_csv(df)
    df = preprocess_text(df)


def get_embeddings(df:pd.DataFrame):
    # Encode les mots concaténés dans le DataFrame
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedings = model.encode(df["Concatenated Words"].tolist())
    df_result = pd.DataFrame()
    df_result["Marque"] = df["Marque"]
    df_result["Embeddings"] = embedings.tolist()
    return df_result

def save_embeddings(df:pd.DataFrame):
    # Sauvegarder les embeddings dans un fichier csv
    df_result = get_embeddings(df)
    df_result.to_csv("embeddings.csv", sep=';', encoding='UTF-8', index=False)


def get_embedding_description(l:list[str]):
    # Prend une liste de chaînes de caractères en entrée, utilise le modèle SentenceTransformer('all-MiniLM-L6-v2') pour encoder les chaînes de caractères, et renvoie les embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(l)


if __name__ == '__main__':
    get_data()