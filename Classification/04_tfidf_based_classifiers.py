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

data = pd.read_csv("/home/jupyter-ozkan_ma/data/nela_kaggle.csv")
print("Number of NULL-Values:")
print(data.isna().sum())
data.dropna(inplace=True)
print("Number of NULL-Values:")
print(data.isna().sum())

X = data["content"]
y = data["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42)

tf_vec_unigram = TfidfVectorizer(stop_words="english", min_df=0.333, max_features=2500)

X_train_tf_unigram = tf_vec_unigram.fit_transform(X_train)
X_test_tf_unigram = tf_vec_unigram.transform(X_test)

dt = DecisionTreeClassifier(random_state=42)
svc = LinearSVC()
lr = LogisticRegression(multi_class="multinomial")
nb = BernoulliNB()

classifier = [dt, svc, lr, nb]

for clf in classifier:
    
    print("Training of the classifier: {} \n".format(clf))
    clf.fit(X_train_tf_unigram, y_train)
    y_pred = clf.predict(X_test_tf_unigram)
    
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