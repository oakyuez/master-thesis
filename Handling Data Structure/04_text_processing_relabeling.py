import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

news = pd.read_csv("/home/jupyter-ozkan_ma/data/CSV/nela_kaggle_combined.csv", index_col=None).drop("Unnamed: 0", axis=1)

def remove_punct(text):
    no_punct = "".join([c for c in text if c not in string.punctuation])
    return no_punct

tokenizer = RegexpTokenizer(r'\w+')

def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words("english")]
    return words

porter = PorterStemmer()

def porter_stem(text):
    stem_text = [porter.stem(i) for i in text]
    return stem_text

def list_to_string(text):
    
    listToStr = ' '.join(map(str, text)) 
    return listToStr

def preprocess_text_with_porter(df, old_col, new_col_list, new_col_str):
    df[new_col_list] = df[old_col].apply(lambda x: remove_punct(x))
    df[new_col_list] = df[new_col_list].apply(lambda x: tokenizer.tokenize(x.lower()))
    #df[new_col] = df[new_col].apply(lambda x: remove_stopwords(x))
    df[new_col_list] = df[new_col_list].apply(lambda x: porter_stem(x))
    df[new_col_str] = df[new_col_list].apply(lambda x: list_to_string(x))
    return df

news = preprocess_text_with_porter(news, old_col="content", new_col_list="pre_content_list", new_col_str="pre_content_str")

news["Length"] = news.content.str.split().str.len()

news.loc[news["Label"]=="Left", "Label_AbStudy_01"] = "Left"
news.loc[news["Label"]=="Lean Left", "Label_AbStudy_01"] = "Left"
news.loc[news["Label"]=="Center", "Label_AbStudy_01"] = "Center"
news.loc[news["Label"]=="Lean Right", "Label_AbStudy_01"] = "Right"
news.loc[news["Label"]=="Right", "Label_AbStudy_01"] = "Right"

news.loc[news["Label"]=="Left", "Label_AbStudy_02"] = "Left"
news.loc[news["Label"]=="Lean Left", "Label_AbStudy_02"] = "Center"
news.loc[news["Label"]=="Center", "Label_AbStudy_02"] = "Center"
news.loc[news["Label"]=="Lean Right", "Label_AbStudy_02"] = "Center"
news.loc[news["Label"]=="Right", "Label_AbStudy_02"] = "Right"

news.loc[news["Label"]=="Left", "Label_AbStudy_03"] = "Partisan"
news.loc[news["Label"]=="Lean Left", "Label_AbStudy_03"] = "NonPartisan"
news.loc[news["Label"]=="Center", "Label_AbStudy_03"] = "NonPartisan"
news.loc[news["Label"]=="Lean Right", "Label_AbStudy_03"] = "NonPartisan"
news.loc[news["Label"]=="Right", "Label_AbStudy_03"] = "Partisan"

news.to_csv("/home/jupyter-ozkan_ma/data/CSV/news_preprocessed_with_addtionalLabel.csv")