import time
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

train = pd.concat([split_df_in_train_test(left_FE)[0], \
    split_df_in_train_test(leanLeft_FE)[0], \
    split_df_in_train_test(center_FE)[0], \
    split_df_in_train_test(leanRight_FE)[0], \
    split_df_in_train_test(right_FE)[0]])

test =  pd.concat([split_df_in_train_test(left_FE)[1], \
    split_df_in_train_test(leanLeft_FE)[1], \
    split_df_in_train_test(center_FE)[1], \
    split_df_in_train_test(leanRight_FE)[1], \
    split_df_in_train_test(right_FE)[1]])

X_train, y_train = train["pre_content_str"], train["Label"]
X_test, y_test = test["pre_content_str"], test["Label"]

## Generate TFIDF vector

### Bigram:

bigram_vec = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))

X_train_bi = bigram_vec.fit_transform(X_train.apply(lambda x: np.str_(x)))
X_test_bi = bigram_vec.transform(X_test.apply(lambda x: np.str_(x)))

### Trigram

trigram_vec = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 3))

X_train_tri = trigram_vec.fit_transform(X_train.apply(lambda x: np.str_(x)))
X_test_tri = trigram_vec.transform(X_test.apply(lambda x: np.str_(x)))

## Generate LabelEncoder

label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc = label_enc.fit_transform(y_test)

label_enc.inverse_transform([0, 1, 2, 3, 4]) 

label = [0, 1, 2, 3, 4]
target_label = ["Center", "Lean Left", "Lean Right", "Left", "Right"]

## Apply classifiers
def run_classifier(clf, X_train, X_test, y_train, y_test, label, target_label):
    
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time() - start
    
    print("Duration of training the model in sec: {} \n".format(end))
    
    start = time.time()
    y_pred = clf.predict(X_test)
    end = time.time() - start
    
    print("Duration of applying the model to unseen data in sec: {} \n".format(end))

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

dt = DecisionTreeClassifier(random_state=42)
svc = LinearSVC()
lr = LogisticRegression(multi_class="multinomial", solver="saga")
nb = BernoulliNB()

### ...using with bigrams

#### Decision Tree

run_classifier(dt, X_train_bi, X_test_bi, y_train_enc, y_test_enc, label, target_label)

#### Naive Bayes

run_classifier(nb, X_train_bi, X_test_bi, y_train_enc, y_test_enc, label, target_label)

#### Support Vector Machine

run_classifier(svc, X_train_bi, X_test_bi, y_train_enc, y_test_enc, label, target_label)

#### Logistic Regression

run_classifier(lr, X_train_bi, X_test_bi, y_train_enc, y_test_enc, label, target_label)

### ...using trigrams

#### Decision Tree

run_classifier(dt, X_train_tri, X_test_tri, y_train_enc, y_test_enc, label, target_label)

#### Naive Bayes

run_classifier(nb, X_train_tri, X_test_tri, y_train_enc, y_test_enc, label, target_label)

#### Support Vector Machine

run_classifier(svc, X_train_tri, X_test_tri, y_train_enc, y_test_enc, label, target_label)

#### Logistic Regression

run_classifier(lr, X_train_tri, X_test_tri, y_train_enc, y_test_enc, label, target_label)

