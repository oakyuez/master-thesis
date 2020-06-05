import pandas as pd
import sqlite3

# Import NELA 2018 database and convert to csv-file 
con_nela =  sqlite3.connect("C:/Users/Oezkan/Desktop/Thesis/Data/NELA/articles.db")

nela = pd.read_sql_query("Select * From articles", con_nela)

nela.to_csv("C:/Users/Oezkan/Desktop/Thesis/Data/csv_files/nela.csv")

# Import Kaggle database and convert to csv-file
con_kaggle = sqlite3.connect("C:/Users/Oezkan/Desktop/Thesis/Data/Kaggle/all-the-news.db")

kaggle = pd.read_sql_query("Select * from longform", con_kaggle)

kaggle.to_csv("C:/Users/Oezkan/Desktop/Data/csv_files/kaggle.csv")
