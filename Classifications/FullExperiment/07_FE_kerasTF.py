import keras 
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, SimpleRNN, GRU, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import string
import re
import matplotlib.pyplot as plt

## Load txt files

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/train/",
                                                          labels="inferred",
                                                          label_mode="categorical",
                                                          batch_size=64,
                                                          validation_split=0.2,
                                                          subset="training",        
                                                          seed=1337)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/train/",
                                                          labels="inferred",
                                                          label_mode="categorical",
                                                          batch_size=64,
                                                          validation_split=0.2,
                                                          subset="validation",
                                                          seed=1337)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/data/TXT/Full_Experiment/test/",
                                                         labels="inferred",
                                                         label_mode="categorical",
                                                         batch_size=64)

### Model Constants

max_features = 30000
embedding_dim = 128
sequence_length = 512
epochs=10

### Text Vectorizer

vectorize_layer = TextVectorization(max_tokens=max_features,
                                    standardize="lower_and_strip_punctuation",
                                    output_mode="int",
                                    output_sequence_length=sequence_length)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

## Function to plotting accuracy and loss

def plot_train_val_loss(epochs, loss, val_loss):
    
    plt.clf()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def plot_train_val_acc(epochs, acc, val_acc):
    
    plt.clf()   # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

## Function to compile and train model, evaluate its performance on the test set

def run_model(input_layer, output_layer, train_data, val_data, test_data):
    
    model = tf.keras.Model(input_layer, output_layer)
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print("Training of the model: \n")
    model_history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=callback)
    
    print("Performance of the model to unseen data: \n")
    model.evaluate(test_data)
    
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs_x = range(1, len(acc) + 1)
    
    plot_train_val_acc(epochs_x, acc, val_acc)
    plot_train_val_loss(epochs_x, loss, val_loss)

## Core layers and arguments for reusing

# A integer input for vocab indices.
input_layer = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
embedding_layer = Embedding(max_features, embedding_dim)(input_layer)

# We project onto five unit output layer, and squash it with a softmax:
# output_layer = Dense(5, activation="softmax", name="predictions")

callback = EarlyStopping(monitor="val_loss", patience=2)

## 9. Model: mod_09

mod09_lay01 = Dropout(0.5)(embedding_layer)
mod09_lay02 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod09_lay01)
mod09_lay03 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod09_lay02)
mod09_lay04 = GlobalMaxPooling1D()(mod09_lay03)
mod09_lay05 = Dense(128, activation="relu")(mod09_lay04)
mod09_lay06 = Dropout(0.5)(mod09_lay05)

mod09_outputLayer = Dense(5, activation="softmax")(mod09_lay06)

run_model(input_layer, mod09_outputLayer, train_ds, val_ds,test_ds)

## 10. Model: mod_10

mod10_lay01 = GlobalMaxPooling1D()(embedding_layer)
mod10_lay02 = Dense(64, activation="relu")(mod10_lay01)
mod10_lay03 = Dense(32, activation="relu")(mod10_lay02)      
mod10_lay04 = Dense(16, activation="relu")(mod10_lay03)

mod10_outputLayer = Dense(5, activation="softmax")(mod10_lay04)

run_model(input_layer, mod10_outputLayer, train_ds, val_ds,test_ds)

## 11. Model: mod_11

mod11_lay01 = Dropout(0.5)(embedding_layer)
mod11_lay02 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod11_lay01)
mod11_lay03 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod11_lay02)
mod11_lay04 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod11_lay03)
mod11_lay05 = GlobalMaxPooling1D()(mod11_lay04)

mod11_outputLayer = Dense(5, activation="softmax")(mod11_lay05)

run_model(input_layer, mod11_outputLayer, train_ds, val_ds,test_ds)

## 12. Model: mod_12

mod12_lay01 = GRU(128)(embedding_layer)
mod12_lay02 = Dense(64, activation="relu")(mod12_lay01)
mod12_lay03 = Dense(16, activation="relu")(mod12_lay02)

mod12_outputLayer = Dense(5, activation="softmax")(mod12_lay03)

run_model(input_layer, mod12_outputLayer, train_ds, val_ds,test_ds)

## 13. Model: mod_13

mod13_lay01 = Bidirectional(GRU(128, return_sequences=True))(embedding_layer)
mod13_lay02 = Bidirectional(GRU(64, return_sequences=True))(mod13_lay01)
mod13_lay03 = Bidirectional(GRU(16))(mod13_lay02)

mod13_outputLayer = Dense(5, activation="softmax")(mod13_lay03)

run_model(input_layer, mod13_outputLayer, train_ds, val_ds,test_ds)

## 14. Model: mod_14

mod14_lay01 = LSTM(148)(embedding_layer)
mod14_lay02 = Dense(64, activation="relu")(mod14_lay01)
mod14_lay03 = Dense(16, activation="relu")(mod14_lay02)

mod14_outputLayer = Dense(5, activation="softmax")(mod14_lay03)

run_model(input_layer, mod14_outputLayer, train_ds, val_ds,test_ds)

## 15. Model: mod_15

mod15_lay01 = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
mod15_lay02 = Bidirectional(LSTM(64, return_sequences=True))(mod15_lay01)
mod15_lay03 = Bidirectional(LSTM(16))(mod15_lay02)

mod15_outputLayer = Dense(5, activation="softmax")(mod15_lay03)

run_model(input_layer, mod15_outputLayer, train_ds, val_ds,test_ds)

## 16. Model: mod_16

mod16_lay01 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(embedding_layer)
mod16_lay02 = Conv1D(128, 7, padding="valid", activation="relu", strides=3)(mod16_lay01)
mod16_lay03 = Bidirectional(LSTM(128, return_sequences=True, go_backwards=True))(mod16_lay02)
mod16_lay04 = Bidirectional(LSTM(64))(mod16_lay03)
mod16_lay05 = Dense(16, activation="relu")(mod16_lay04)

mod16_outputLayer = Dense(5, activation="softmax")(mod16_lay05)

run_model(input_layer, mod16_outputLayer, train_ds, val_ds,test_ds)

