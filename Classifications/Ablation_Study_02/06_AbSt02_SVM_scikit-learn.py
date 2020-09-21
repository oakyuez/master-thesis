import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle, resample

## Load data

news = pd.read_csv("/home/jupyter-ozkan_ma/data/CSV/news_preprocessed_with_addtionalLabel.csv", index_col=0)

# Get the same train and test data
def split_df_in_train_test(df):
    df = df.reset_index()
    split_point = int(np.round(df.shape[0]) * 0.8)
    df_train = df.loc[:split_point-1,:]
    df_test = df.loc[split_point:,:]
    return df_train, df_test

left_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
center_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Center") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
right_AbSt_02 = resample(shuffle(news[(news["Label_AbStudy_02"]=="Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)

train = pd.concat([split_df_in_train_test(left_AbSt_02)[0], \
    split_df_in_train_test(center_AbSt_02)[0], \
    split_df_in_train_test(right_AbSt_02)[0]])

test = pd.concat([split_df_in_train_test(left_AbSt_02)[1], \
    split_df_in_train_test(center_AbSt_02)[1], \
    split_df_in_train_test(right_AbSt_02)[1]])

X_train, y_train = train["pre_content_str"], train["Label_AbStudy_02"]
X_test, y_test = test["pre_content_str"], test["Label_AbStudy_02"]

## Generate trigram tfidf vector

trigram_vec = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 3))

X_train_tri = trigram_vec.fit_transform(X_train.apply(lambda x: np.str_(x)))
X_test_tri = trigram_vec.transform(X_test.apply(lambda x: np.str_(x)))

## Generate Label Encoder

label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc = label_enc.fit_transform(y_test)

label_enc.inverse_transform([0, 1, 2]) 

label = [0, 1, 2]
target_label = ["Center", "Left", "Right"]

## Define function to run and evaluate classifier

def run_classifier(clf, X_train, X_test, y_train, y_test, label, target_label):
    
    print("Training of the classifier: {} \n".format(clf))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n")

    print("Accuracy of the classifier:     ")
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    print("\n")

    print("Confusion Matrix of the classifier: \n")
    con_mat = confusion_matrix(y_test, y_pred, labels=label)
    print(con_mat)

    print("\n")

    print("Classification Report of the classifier: \n")
    report = classification_report(y_test, y_pred, target_names=target_label)
    print(report)

## Train a SVM

svc = LinearSVC()

run_classifier(svc, X_train_tri, X_test_tri, y_train_enc, y_test_enc, label, target_label)

