import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Reading csv files for each target label
left = pd.read_csv("C:/Datasets/Ready to Classify/news_left.csv")
lean_left = pd.read_csv("C:/Datasets/Ready to Classify/news_leanLeft.csv")
center = pd.read_csv("C:/Datasets/Ready to Classify/news_Center.csv")
lean_right = pd.read_csv("C:/Datasets/Ready to Classify/news_leanRight.csv")
right = pd.read_csv("C:/Datasets/Ready to Classify/news_Right.csv")

#print("Number of rows of left dataframe: {}".format(left.shape[0]))
#print("Number of rows of lean left dataframe: {}".format(lean_left.shape[0]))
#print("Number of rows of center dataframe: {}".format(center.shape[0]))
#print("Number of rows of lean right dataframe: {}".format(lean_right.shape[0]))
#print("Number of rows of right dataframe: {}".format(right.shape[0]))

# Permutate the dataframes
df_list = [left, lean_left, center, lean_right, right]

for df in df_list:
    df = shuffle(df)

# Function to split the dataframe in train and test set without separating into feature and target column like scikit-learn
def split_df_in_train_test(df):
    split_point = int(np.round(df.shape[0]) * 0.8)
    df_train = df.iloc[:split_point, 2]
    df_test = df.iloc[split_point:, 2]
    return df_train, df_test

# Function to write one row (news) per dataframe as a txt.file
def news_df_to_txt(dataframe, train_directory, test_directory, label):
    
    train, test = split_df_in_train_test(dataframe)
    
    for index, value in train.iteritems():
        name = train_directory+label+"/"+label+"_news_nr_{}.txt".format(index)
        file = open(name, mode="w", encoding="utf8")
        file.write(value)
        
    for index, value in test.iteritems():
        name = test_directory+label+"/"+label+"_news_nr_{}.txt".format(index)
        file = open(name, mode="w", encoding="utf8")
        file.write(value)
        
    file.close()

# Convert the dataframe into txt file
news_df_to_txt(left, 
                train_directory="C:/Datasets/Labeled/Train/", 
                test_directory="C:/Datasets/Labeled/Test/", 
                label="Left")

news_df_to_txt(lean_left, 
                train_directory="C:/Datasets/Labeled/Train/", 
                test_directory="C:/Datasets/Labeled/Test/", 
                label="LeanLeft")

news_df_to_txt(center, 
                train_directory="C:/Datasets/Labeled/Train/", 
                test_directory="C:/Datasets/Labeled/Test/", 
                label="Center")

news_df_to_txt(lean_right, 
                train_directory="C:/Datasets/Labeled/Train/", 
                test_directory="C:/Datasets/Labeled/Test/", 
                label="LeanRight")

news_df_to_txt(right, 
                train_directory="C:/Datasets/Labeled/Train/", 
                test_directory="C:/Datasets/Labeled/Test/", 
                label="Right")                                                                