pip install ktrain

import ktrain
from ktrain import text
import pandas as pd
import numpy as np
from sklearn.utils import shuffle, resample

Load the data

news = pd.read_csv("/content/drive/My Drive/CSV/news_preprocessed_with_addtionalLabel.csv", index_col=0)

# Get the same train and test data
def split_df_in_train_test(df):
    df = df.reset_index()
    split_point = int(np.round(df.shape[0]) * 0.8)
    df_train = df.loc[:split_point-1,:]
    df_test = df.loc[split_point:,:]
    return df_train, df_test

left_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Left") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
center_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Center") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)
right_AbSt_01 = resample(shuffle(news[(news["Label_AbStudy_01"]=="Right") & (news["Length"]<512)], random_state=42), \
         random_state=42, n_samples=25000)

train = pd.concat([split_df_in_train_test(left_AbSt_01)[0], \
    split_df_in_train_test(center_AbSt_01)[0], \
    split_df_in_train_test(right_AbSt_01)[0]])

test = pd.concat([split_df_in_train_test(left_AbSt_01)[1], \
    split_df_in_train_test(center_AbSt_01)[1], \
    split_df_in_train_test(right_AbSt_01)[1]])

x_train = train["content"].to_list()
y_train = train["Label_AbStudy_01"].to_list()
x_test = test["content"].to_list()
y_test = test["Label_AbStudy_01"].to_list()

class_names = list(train["Label_AbStudy_01"].unique())

Apply the model

model_name = "roberta-base"

t = text.Transformer(model_name, maxlen=512, class_names=class_names)

trn = t.preprocess_train(x_train, y_train)
val = t.preprocess_test(x_test, y_test)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6)

learner.lr_find(show_plot=True, max_epochs=1)

optimal_lr = learner.lr_estimate()[1]
print(optimal_lr)

learner.fit(optimal_lr, 1)

learner.validate(class_names=class_names)

