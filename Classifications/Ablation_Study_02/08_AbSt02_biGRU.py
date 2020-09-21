import ktrain
from ktrain import text

## Load data

trn, val, preproc = text.texts_from_folder("/home/jupyter-ozkan_ma/data/TXT/Ablation_Study_02/", 
                                           max_features=20000, maxlen=512, 
                                           ngram_range=1, 
                                           preprocess_mode='standard',
                                           classes=['Center', 'Left', 'Right'])

## Inspection of available models

text.print_text_classifiers()

## Apply the bigru model

bigru = text.text_classifier("bigru", trn, preproc=preproc)

learner_bigru = ktrain.get_learner(bigru, train_data=trn, val_data=val)

learner_bigru.lr_find(show_plot=True, max_epochs=5)

learner_bigru.lr_estimate()

learner_bigru.fit(learner_bigru.lr_estimate()[1], 5)