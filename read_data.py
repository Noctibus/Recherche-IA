import pandas as pd
import re
from nltk.corpus import stopwords

def read_csv ():
    return pd.read_csv("data.csv", sep=';', encoding='UTF-8')

def preprocess_csv (df:pd.DataFrame):
    # Extraction des marques avec plusieurs noms / description
    df['Brand'] = df['Marque'].str.extract(r'"([^"]*)')
    df.loc[~df['Brand'].isnull(), 'Marque'] = df['Brand']

    # Traitement des caractères spéciaux
    df['Marque'] = df['Marque'].apply(lambda x: re.sub(r'[^a-zA-ZÀ-ÖØ-öø-ÿ\s]', '', str(x)))

    # Drop si le nom est vide après traitement
    df = df.dropna(subset=['Marque'])
    df = df[df['Marque'].str.strip() != '']

    return df


def preprocess_text(df:pd.DataFrame):
    # remove articles from the column "Produits et services", like "le", "la", "les", "l'", "un", "une", "des", "du", "de", "d'"
    df["Produits et services"] = df["Produits et services"].apply(lambda x: re.sub(r"\b(le|la|les|l'|un|une|des|du|de|d')\b", "", x))

    # split the column "Produits et services" into a list of words with the separator ";"
    df["Produits et services"] = df["Produits et services"].apply(lambda x: x.split(";"))

    # remove characters except spaces that are not letters for each element of the list inside the column "Produits et services"
    df["Produits et services"] = df["Produits et services"].apply(lambda x: [re.sub(r"[^a-zA-ZÀ-ÖØ-öø-ÿ\s ]+", "", i) for i in x])

    # convert to lowercase and split into words
    df["Produits et services"] = df["Produits et services"].apply(lambda x: [i.lower().split() for i in x])

    # remove stopwords
    stops = set(stopwords.words("french"))
    df["Produits et services"] = df["Produits et services"].apply(lambda x: [[j for j in i if j not in stops] for i in x])

    # add a new column : it is the concatenation of the column "Marque" and "Produits et services"
    df["Words list"] = df.apply(lambda row: [[row["Marque"]]] + row["Produits et services"], axis=1)

    # the column "Word list" is like [["hello", "world"], ["good", "morning"]]. Convert it to ["hello", "world", "good", "morning"] into a new column "Concatenated Words"
    df["Concatenated Words"] = df["Produits et services"].apply(lambda x: [j for i in x for j in i])
    df["Concatenated Words"] = df.apply(lambda row: [row["Marque"]] + row["Concatenated Words"], axis=1)

    return df