import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Reading csv files for each target label
left = pd.read_csv("/home/jupyter-ozkan_ma/data/news_Left.csv")
lean_left = pd.read_csv("/home/jupyter-ozkan_ma/data/news_leanLeft.csv")
center = pd.read_csv("/home/jupyter-ozkan_ma/data/news_Center.csv")
lean_right = pd.read_csv("/home/jupyter-ozkan_ma/data/news_leanRight.csv")
right = pd.read_csv("/home/jupyter-ozkan_ma/data/news_Right.csv")

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

# Function to split the dataframe in train and test set a feature and target column like scikit-learn
def split_df_in_train_test_with_label(df):
    split_point = int(np.round(df.shape[0]) * 0.8)
    df_train = df.iloc[:split_point, 2:]
    df_test = df.iloc[split_point:, 2:]
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

## Generating a dataframe containing the training dataset 

train_leanLeft = split_df_in_train_test_with_label(lean_left)[0]
train_left = split_df_in_train_test_with_label(left)[0]
train_center = split_df_in_train_test_with_label(center)[0]
train_leanRight = split_df_in_train_test_with_label(lean_right)[0]
train_right = split_df_in_train_test_with_label(right)[0]

train_df = train_leanLeft.append(train_left).append(train_center).append(train_leanRight).append(train_right)

## Save the training dataset as a csv file for later classification
train_df.to_csv("/home/jupyter-ozkan_ma/data/CSV/training_data.csv")

## Generating a dataframe containing the test dataset 

test_leanLeft = split_df_in_train_test_with_label(lean_left)[1]
test_left = split_df_in_train_test_with_label(left)[1]
test_center = split_df_in_train_test_with_label(center)[1]
test_leanRight = split_df_in_train_test_with_label(lean_right)[1]
test_right = split_df_in_train_test_with_label(right)[1]

test_df = test_leanLeft.append(test_left).append(test_center).append(test_leanRight).append(test_right)

## Save the test dataset as a csv file for later classification
test_df.to_csv("/home/jupyter-ozkan_ma/data/CSV/test_data.csv")

# Specifying the directory the store the txt file 

train_dir_txt = "/home/jupyter-ozkan_ma/data/TXT/Train/"
test_dir_txt = "/home/jupyter-ozkan_ma/data/TXT/Test/"

## Generating a txt file for each news in the dataframes

news_df_to_txt(left, 
                train_directory=train_dir_txt, 
                test_directory=test_dir_txt, 
                label="Left")

news_df_to_txt(lean_left, 
                train_directory=train_dir_txt, 
                test_directory=test_dir_txt, 
                label="LeanLeft")

news_df_to_txt(center, 
                train_directory=train_dir_txt, 
                test_directory=test_dir_txt, 
                label="Center")

news_df_to_txt(lean_right, 
                train_directory=train_dir_txt, 
                test_directory=test_dir_txt, 
                label="LeanRight")

news_df_to_txt(right, 
                train_directory=train_dir_txt, 
                test_directory=test_dir_txt, 
                label="Right")                      