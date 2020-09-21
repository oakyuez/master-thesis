import ktrain
from ktrain import text

## Loading data

trn, val, preproc = text.texts_from_folder("/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/", 
                                           max_features=20000, maxlen=512, 
                                           ngram_range=1, 
                                           preprocess_mode='standard',
                                           classes=['Center', 'LeanLeft', 'LeanRight', 'Left', 'Right'])

## Inspection of available classifiers

text.print_text_classifiers()

### Applying the fasttext model (mod_17):

fasttext = text.text_classifier("fasttext", trn, preproc=preproc)

learner_ft = ktrain.get_learner(fasttext, train_data=trn, val_data=val)

learner_ft.lr_find(show_plot=True, max_epochs=5)

learner_ft.lr_estimate()

learner_ft.fit(learner_ft.lr_estimate()[1], 5)

# Since val_loss still decreass train for 5 epochs
learner_ft.fit(learner_ft.lr_estimate()[1], 5)

# Since val_loss still decreass train for 5 epochs
learner_ft.fit(learner_ft.lr_estimate()[1], 5)

# Since val_loss still decreass train for 5 epochs
learner_ft.fit(learner_ft.lr_estimate()[1], 5)

## Applying the logreg model: (mod_18)

logreg = text.text_classifier("logreg", trn, preproc=preproc)

learner_log = ktrain.get_learner(logreg, train_data=trn, val_data=val)

learner_log.lr_find(show_plot=True, max_epochs=10)

learner_log.lr_estimate()

learner_log.fit(learner_log.lr_estimate()[1], 10)

## Applying the nbsvm model: (mod_19)

nbsvm = text.text_classifier("nbsvm", trn, preproc=preproc)

learner_nbsvm = ktrain.get_learner(nbsvm, train_data=trn, val_data=val)

learner_nbsvm.lr_find(show_plot=True, max_epochs=10)

learner_nbsvm.lr_estimate()

learner_nbsvm.fit(learner_nbsvm.lr_estimate()[1], 10)

## Applying the bigru model: (mod_20)

bigru = text.text_classifier("bigru", trn, preproc=preproc)

learner_bigru = ktrain.get_learner(bigru, train_data=trn, val_data=val)

learner_bigru.lr_find(show_plot=True, max_epochs=10)

learner_bigru.lr_estimate()

learner_bigru.fit(learner_bigru.lr_estimate()[1], 7)

## Applying the standard gru: (mod_21)

stgru = text.text_classifier("standard_gru", trn, preproc=preproc)

learner_stgru = ktrain.get_learner(stgru, train_data=trn, val_data=val)

learner_stgru.lr_find(show_plot=True, max_epochs=10)

learner_stgru.lr_estimate()

learner_stgru.fit(learner_stgru.lr_estimate()[1], 7)

