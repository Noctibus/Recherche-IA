import pandas as pd
from read_data import read_csv, preprocess_csv, preprocess_text
from sentence_transformers import SentenceTransformer

df = read_csv()
df = preprocess_csv(df)
df = preprocess_text(df)

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(df:pd.DataFrame):
    embedings = model.encode(df["Concatenated Words"].tolist())
    df_result = pd.DataFrame()
    df_result["Marque"] = df["Marque"]
    df_result["Embeddings"] = embedings.tolist()
    return df_result

df_result = get_embeddings(df)
df_result.to_csv("embeddings.csv", sep=';', encoding='UTF-8', index=False)