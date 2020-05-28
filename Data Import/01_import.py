import pandas as pd

# Import Nela Kaggle-News datasets and keep only columns regarding the title and source>
nela = pd.read_csv("C:/Datasets/_all_features_NELA2017dataset.csv")

# Import Nela 2018 and 2019 db into an dataframe using this sample code

'''
import pandas as pd
import sqlite3

# Read sqlite query results into a pandas DataFrame
con = sqlite3.connect("data/portal_mammals.sqlite")
df = pd.read_sql_query("SELECT * from surveys", con)

# Verify that result of SQL query is stored in the dataframe
print(df.head())

con.close()
'''


# Import Kaggle-News datasets, concat them and keep only columns like the title and source>
kaggle_1 = pd.read_csv("C:/Datasets/articles1.csv")
kaggle_2 = pd.read_csv("C:/Datasets/articles2.csv")
kaggle_3 = pd.read_csv("C:/Datasets/articles3.csv")

kaggle = pd.concat([kaggle_1, kaggle_2, kaggle_3], axis=0)

kaggle = kaggle.loc[: ,["title", "publication"]]

# Concat both dataframes 
news = pd.concat()

# Save dataframe as csv
news.to_csv()
