import numpy as np
import pandas as pd
from sklearn.utils import shuffle, resample

news = pd.read_csv("/home/jupyter-ozkan_ma/data/CSV/news_preprocessed_with_addtionalLabel.csv", index_col=0)

## Define function to txt file for each news

# Function to split the dataframe in train and test set without separating into feature and target column like scikit-learn
def split_df_in_train_test(df):
    df = df.reset_index()
    split_point = int(np.round(df.shape[0]) * 0.8)
    df_train = df.loc[:split_point-1, "content"]
    df_test = df.loc[split_point:, "content"]
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

## Generate txt files for the full experiment

left_FE = resample(shuffle(news[(news["Label"]=="Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=15000)
leanLeft_FE = resample(shuffle(news[(news["Label"]=="Lean Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=15000)
center_FE = resample(shuffle(news[(news["Label"]=="Center") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=15000)
leanRight_FE = resample(shuffle(news[(news["Label"]=="Lean Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=15000)
right_FE = resample(shuffle(news[(news["Label"]=="Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=15000)

FE_train_directory = "/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/train/"
FE_test_directory = "/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/test/"

news_df_to_txt(left_FE, FE_train_directory, FE_test_directory, "Left")
news_df_to_txt(leanLeft_FE, FE_train_directory, FE_test_directory, "LeanLeft")
news_df_to_txt(center_FE, FE_train_directory, FE_test_directory, "Center")
news_df_to_txt(leanRight_FE, FE_train_directory, FE_test_directory, "LeanRight")
news_df_to_txt(right_FE, FE_train_directory, FE_test_directory, "Right")

## Generate txt file for the first ablation study

left_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
center_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Center") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
right_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)

abSt01_train_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_01/train/"
abSt01_test_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_01/test/"

news_df_to_txt(left_AbSt_01, abSt01_train_directory, abSt01_test_directory, "Left")
news_df_to_txt(center_AbSt_01, abSt01_train_directory, abSt01_test_directory, "Center")
news_df_to_txt(right_AbSt_01, abSt01_train_directory, abSt01_test_directory, "Right")

## Generate txt files for the second ablation study

left_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
center_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Center") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
right_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)

abSt02_train_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_02/train/"
abSt02_test_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_02/test/"

news_df_to_txt(left_AbSt_02, abSt02_train_directory, abSt02_test_directory, "Left")
news_df_to_txt(center_AbSt_02, abSt02_train_directory, abSt02_test_directory, "Center")
news_df_to_txt(right_AbSt_02, abSt02_train_directory, abSt02_test_directory, "Right")

## Generate txt files for the third ablation study

partisanNews = resample(shuffle(news[(news["Label_AbStudy_03"]=="Partisan") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=37500)
nonPartisanNews = resample(shuffle(news[(news["Label_AbStudy_03"]=="NonPartisan") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=37500)

abSt03_train_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_03/train/"
abSt03_test_directory = "/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_03/test/"

news_df_to_txt(partisanNews, abSt03_train_directory, abSt03_test_directory, "Partisan")
news_df_to_txt(nonPartisanNews, abSt03_train_directory, abSt03_test_directory, "NonPartisan")

