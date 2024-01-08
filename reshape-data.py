import pandas as pd
from pprint import pprint

# read Data-A-Z.csv file. The header is N° de la marque;Marque;Type de la marque;Date de dépôt/enregistrement;Pays de priorité;Date de priorité;n° de priorité;"Produits et services"
df_a_z = pd.read_csv('Data-A-Z.csv', sep=';', encoding='UTF-8')
df_z_a = pd.read_csv('Data-Z-A.csv', sep=';', encoding='UTF-8')
df_ancien_recent = pd.read_csv('Data-ancien-recent.csv', sep=';', encoding='UTF-8')
df_recent_ancien = pd.read_csv('Data-recent-ancien.csv', sep=';', encoding='UTF-8')

df = pd.concat([df_a_z, df_z_a, df_ancien_recent, df_recent_ancien], ignore_index=True)

# remove duplicates
df = df.drop_duplicates(subset=['Marque', 'Type de la marque', 'Date de dépôt/enregistrement', 'Pays de priorité', 'Date de priorité', 'n° de priorité', 'Produits et services'], keep='first')

print(df.shape)

# remove the columns 'Pays de priorité', 'Date de priorité', 'n° de priorité'
df = df.drop(['Pays de priorité', 'Date de priorité', 'n° de priorité'], axis=1)

print(df.head())

# export to csv file
df.to_csv('data.csv', sep=';', encoding='UTF-8', index=False)