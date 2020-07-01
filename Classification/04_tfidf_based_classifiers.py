import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

## MultiClass Classifiers from scikit-learn: https://scikit-learn.org/stable/modules/multiclass.html#multiclass
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("/home/jupyter-ozkan_ma/data/nela_kaggle_combined.csv")
print("Number of NULL-Values:")
print(data.isna().sum())

X = data["content"]
y = data["Label"]

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)

label_enc.inverse_transform([0, 1, 2, 3 , 4])
label = [0, 1, 2, 3, 4]
target_label = ["Center", "Lean Left", "Lean Right", "Left", "Right"]

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
svc = LinearSVC()
lr = LogisticRegression(multi_class="multinomial")
nb = BernoulliNB()

classifier = [dt, svc, lr, nb]

for i in range(1, 4):
    
    print("Training of the classifiers with n-gram range from 1 to {}: \n".format(i))
    
    tfidf_vec = TfidfVectorizer(stop_words="english", min_df=0.33, max_features=1000, ngram_range=(1, i))
    
    X_train_tfidf = tfidf_vec.fit_transform(X_train)
    X_test_tfidf = tfidf_vec.transform(X_test)
    
    for clf in classifier:
    
        print("Training of the classifier: {} \n".format(clf))
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)

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

        print("\n")
        