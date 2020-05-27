import pandas as pd

# Import Nela Kaggle-News datasets and keep only columns regarding the title and source>
nela = pd.read_csv("C:/Datasets/_all_features_NELA2017dataset.csv")

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
