import pandas as pd
from read_data import read_csv, preprocess_csv, preprocess_text

df = read_csv()
df = preprocess_csv(df)
df = preprocess_text(df)

print(df["Words list"][10])
print(df["Concatenated Words"][10])