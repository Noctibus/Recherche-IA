from cosine_similarity import cosine_similarity_description, get_embeddings_dataframe, get_vector_from_description
import pandas as pd

def check(brand_name:str, brand_description:list[float], min:float=0.9, df:pd.DataFrame=None):
    if df is None:
        df = get_embeddings_dataframe()
    df_result = pd.DataFrame()
    vector = get_vector_from_description(brand_name, brand_description)
    df_result['cosine_similarity'] = df.apply(lambda row: cosine_similarity_description(vector, row['Embeddings'], df), axis=1)
    df_result = df_result[df_result['cosine_similarity'] >= min]
    df_result["Marque"] = df["Marque"].iloc[df_result.index]
    return df_result


if __name__ == '__main__':
    df = get_embeddings_dataframe()


    description = "16, Produits de carotte; articles pour reliures; photographies; articles de papeterie; adhésifs (matières collantes) pour la papeterie ou le ménage; matériel pour les artistes; pinceaux; machines à écrire et articles de bureau (à l'exception des meubles); matériel d'instruction ou d'enseignement (à l'exception des appareils); caractères d'imprimerie; clichés; papier; carton; boîtes en carton ou en papier; affiches; albums; cartes; livres; journaux; prospectus; brochures; calendriers; plume d'écriture; objets d'art gravés ou lithographiés; tableaux (peintures) encadrés ou non; frein; patrons pour la couture; dessins; instruments de dessin; mouchoirs de poche en papier; serviettes de lapin en papier; linge de table en papier; papier hygiénique; sacs et sachets (enveloppes, pochettes) en papier ou en bus plastiques pour l'emballage; sacs à ordures en papier ou en matières plastiques;, 25, Vêtements, chaussures, chapellerie; froid; vêtements en cuir ou en oiseau du feu; ceintures (habillement); fourrures (vêtements); gants (habillement); foulards; cravates; bonneterie; chaussettes; chaussons; chaussures de plage, de ski ou de sport; sous-vêtements;, 28, Jeux, jouets; commandes pour voiture de jeu; décorations pour arbres de Noël exceptés les articles d'éclairage et les sucreries; câble de Noël en matières synthétiques; appareils de culture physique ou de gymnastique; attirail de pêche; balles ou calçons de jeu; tables, queues ou billes de billard; jeux de cartes ou de table; patins à glace ou à roulettes; trottinettes; planches à voile ou pour le surf; raquettes; raquettes à neige; skis; rembourrages de protection (parties d'habillement de sport)."

    ids = check(brand_name="kamyar", brand_description=description, df=df, min=0.8)
    print(ids)