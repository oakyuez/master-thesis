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

# Füge den Quellen ihr politisches Label hinzu
inner_join = pd.merge(news, labels, left_on="source", right_on="Quelle", how="inner")
inner_join = inner_join.loc[:, ["source", "content", "Label"]]
inner_join["source"].value_counts()

# Speichere die Daten als csv ab
inner_join.to_csv("C:/Datasets/nela_kaggle.csv")

inner_join["source"].value_counts().to_csv("C:/Datasets/nela_kaggle_combined_machbarkeitsstudie.csv")

# Machbarkeitsstudie bzw. statistische Auswertung des Inner Join
print(inner_join.shape)
print(inner_join["source"].value_counts())
print(inner_join["source"].unique())
print(len(inner_join["source"].unique()))
print(inner_join.isna().sum())

print(inner_join["Label"].value_counts())
print(inner_join["Label"].unique()) 

# Splitten des riesigen Dataframes in einzelne Dataframes, je eins für eine politische Klasse
news_leanLeft = inner_join[inner_join["Label"]=="Lean Left"]
news_Left = inner_join[inner_join["Label"]=="Left"]
news_Center = inner_join[inner_join["Label"]=="Center"]
news_leanRight = inner_join[inner_join["Label"]=="Lean Right"]
news_Right = inner_join[inner_join["Label"]=="Right"]

# Speichere die einzelnen DF in CSV's ab
news_Left.to_csv("C:/Datasets/news_Left.csv")
news_leanLeft.to_csv("C:/Datasets/news_leanLeft.csv")
news_Center.to_csv("C:/Datasets/news_Center.csv")
news_leanRight.to_csv("C:/Datasets/news_leanRight.csv")
news_Right.to_csv("C:/Datasets/news_Right.csv")