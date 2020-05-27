import pandas as pd

# Import Nela and Kaggle-News datasets
nela = pd.read_csv("C:/Datasets/_all_features_NELA2017dataset.csv")

kaggle_1 = pd.read_csv("C:/Datasets/articles1.csv")
kaggle_2 = pd.read_csv("C:/Datasets/articles2.csv")
kaggle_3 = pd.read_csv("C:/Datasets/articles3.csv")

kaggle = pd.concat([kaggle_1, kaggle_2, kaggle_3], axis=0)

