import pandas as pd


# retourne le dataframe original mis à jour, et le dataframe crée 
def extract_real_datas (df:pd.DataFrame):
    SEED = 2023
    nb_sample = 200

    # Extraction de nb_sample lignes
    df_test = df.sample(n=nb_sample, random_state=SEED)

    # Les supprime du dataframe
    df = df.drop(df_test.index)
    
    return df ,df_test




# Donne une liste des antonymes pour chaque entrée
def get_synonyms(words):
    synonyms = set()
    for word in words:
        for syn in wordnet.synsets(word, lang='fra'):
            for lemma in syn.lemmas('fra'):
                synonyms.add(lemma.name())
    return list(synonyms)

# Donne une liste des contraires pour chaque entrée
def get_antonyms_french(words):
    antonyms = set()
    for word in words:
        for syn in wordnet.synsets(word, lang='fra'):
            for lemma in syn.lemmas('fra'):
                for antonym in lemma.antonyms():
                    antonyms.add(antonym.name())
    return list(antonyms)


def create_false_datas (df:pd.DataFrame):
    df_test = df.sample(n=nb_sample, random_state=SEED)
    # TODO : vérifier liste synonymes / antonymes si rien existe => substitution / sinon choix de 1 des deux si existe both

