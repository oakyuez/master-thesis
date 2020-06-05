import pandas as pd

# Laden der Datensatz aus 01_importData.py
nela = pd.read_csv("C:/Datasets/nela2018.csv", index_col=None, low_memory=False)
kaggle = pd.read_csv("C:/Datasets/kaggle.csv", index_col=None, low_memory=False)

# Laden der csv, welche im Expose die Tabelle 12 ist. Sie enthält nur die Quellen und das politische Label  
labels = pd.read_excel("C:/Datasets/Labels.xlsx", index_col=None)

# Behalte nur den Nachrichteninhalt und die Quelle der Nachricht
kaggle = kaggle.loc[:,["publication", "content"]]
# Ändere den Spaltennamen um, damit beide Datensätze gemerged werden können
kaggle.rename(columns={"publication":"source"}, inplace=True)
# Vereinheitlichen der Spaltennamen wie sie im Kaggle-Datensatz vorkommen
kaggle["source"][kaggle["source"]=="Verge"] = "The Verge"
kaggle["source"][kaggle["source"]=="Atlantic"] = "The Atlantic"
kaggle["source"][kaggle["source"]=="Guardian"] = "The Guardian"
kaggle["source"][kaggle["source"]=="New York Times"] = "The New York Times"

# Behalte nur den Nachrichteninhalt und die Quelle der Nachricht
nela = nela.loc[:, ["source", "content"]]

# Vereinen beide Datemsätze zu einem Datensatz
news = pd.concat([kaggle, nela])

# Füge den Quellen ihr politisches hinzu
inner_join = pd.merge(news, labels, left_on="source", right_on="Quelle", how="inner")
inner_join = inner_join.loc[:, ["source", "content", "Label"]]
inner_join["source"].value_counts()

# Speichere die Daten als csv ab
inner_join.to_csv("C:/Datasets/nela_kaggle.csv")
