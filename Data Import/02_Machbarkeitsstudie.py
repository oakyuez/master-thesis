import pandas as pd 

# Laden der Datensätze, die in 01_importData.py zu csv-Dateien umgewandelt worden sind
kaggle = pd.read_csv("C:/Datasets/kaggle.csv", low_memory=False)
nela = pd.read_csv("C:/Datasets/nela2018.csv", low_memory=False)

# Auswertung der einzelnen DF a) Dimension, b) Anzahl Artikel pro Quelle, c) disjunkte Quellen und d) Anzahl Quellen
kaggle.shape
kaggle["publication"].value_counts()
kaggle["publication"].unique()
len(kaggle["publication"].unique()) 

nela.shape
nela["source"].value_counts()
nela["source"].unique()
len(nela["source"].unique())

# Speichern der Anzahl Artikel pro Quelle für die Machbarkeitsstudie
kaggle["publication"].value_counts().to_csv("C:/Datasets/kaggle_machbarkeitstudie.csv")
nela["source"].value_counts().to_csv("C:/Datasets/nela_machbarkeitstudie.csv")
