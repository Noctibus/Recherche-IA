from cosine_similarity import cosine_similarity_description, get_embeddings_dataframe, get_vector_from_description
import pandas as pd

def check(brand_name:str, brand_description:list[float], min:float=0.9, df:pd.DataFrame=None):
    """
    - extraction des embeddings
    - similarité cosinus avec tous les vecteurs connus de la base
    - liste des vecteurs dont la similarité est supérieure à min

    Args:
        brand_name (str): nom de la marque
        brand_description (list[float]): description de l'activité de la marque
        min (float, optional): score minimum de la similarité cosinus : si calcul supérieur à cette valeur, alors l'algorithme estime que la amrque est frauduleuse. Défaut à 0.9.
        df (pd.DataFrame, optional): dataframe des vecteurs d'embeddings déjà calculés (ceux des marques enregistrées et non frauduleuses). Défaut à None : calculé ou lu dans les fonctions utilisées.

    Returns:
        df (pd.DataFrame, optional): dataframe contenant :
            - le code de la marque similaire
            - la similarité cosinus calculée
            - le nom de la marque simmilaire
    """
    if df is None:
        df = get_embeddings_dataframe()
    df_result = pd.DataFrame()
    vector = get_vector_from_description(brand_name, brand_description)
    df_result['cosine_similarity'] = df.apply(lambda row: cosine_similarity_description(vector, row['Embeddings'], df), axis=1)
    df_result = df_result[df_result['cosine_similarity'] >= min]
    df_result["Marque"] = df["Marque"].iloc[df_result.index]
    return df_result


if __name__ == '__main__':
    # execute following to initiate data and files if needed
    # from reshape-data import reshape_data
    # reshape_data()

    df = get_embeddings_dataframe()

    # read marks from data_to_test.csv and compare them to the saved dataset
    df_test = pd.read_csv('data_to_test.csv', sep=';', encoding='UTF-8')
    for mark in df_test['Marque']:
        description = df_test[df_test['Marque'] == mark]['Produits et services'].values[0]
        ids = check(brand_name=mark, brand_description=description, df=df, min=0.8)
        print(ids)