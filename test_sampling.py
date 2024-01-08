import pandas as pd
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')
import random


# retourne le dataframe original mis à jour, et les dataframes crées
def extract_sample_datas (df:pd.DataFrame):
    SEED = 2023
    nb_sample = 100

    # Extraction de nb_sample lignes pour les vraies données
    df_real = df.sample(n=nb_sample, random_state=SEED)

    # Les supprime du dataframe
    df = df.drop(df_real.index)
    df.to_csv("data_all.csv", sep=';', encoding='UTF-8', index=False)
    # Extraction de nb_sample lignes pour les données à falsifier
    df_false = df.sample(n = nb_sample, random_state=SEED)
    return df_real




# Donne une liste des synonymes pour chaque entrée
def get_synonyms(words):
    synonyms = set()
    for word in words:
        for syn in wordnet.synsets(word, lang='fra'):
            for lemma in syn.lemmas('fra'):
                synonyms.add(lemma.name())
    return list(synonyms)


# Change certains mots dans la liste des mots précédemment concaténés de Produits et Services
def change_words(word_list):
    chance_to_change = 0.25
    for i in range(len(word_list)):
        word = word_list[i]
        if random.random() < chance_to_change:
            synonyms = get_synonyms(word)
            if synonyms:
                word_list[i] = synonyms[0]
    return word_list



# Retourne le dataset de test avec des fausses données générées et les vraies données extraites du dataset
def create_false_datas (df:pd.DataFrame, df_false:pd.DataFrame):
    nb_real = 100
    SEED = 2023
    placeholder_word = 'cacahuette'
    placeholder_date = '31/12/2023'

    df_real['Status'] = 'original'
    df_false['Status'] = 'plagiat'
    
    df_false["Concatenated Words"] = df_false["Concatenated Words"].apply(lambda x: ' '.join(change_words(x)))

    for index, brand in df_false['Marque'].items():
        modified_brand = ''
        for word in brand.split():
            syn_list = get_synonyms(word)
            if syn_list:
                modified_brand += syn_list[0] + ' '
            else:
                modified_brand += word + ' '
        modified_brand = modified_brand.strip()
        if modified_brand == brand:
            words = modified_brand.split()
            words[0] = placeholder_word
            modified_brand = ' '.join(words)

        df_false.at[index, 'Marque'] = modified_brand
        df_false.at[index, 'Date de dépôt/enregistrement'] = placeholder_date
        
    df_concat = pd.concat([df_real, df_false])
    df_concat.to_csv("data_to_test.csv", sep=';', encoding='UTF-8', index=False)
