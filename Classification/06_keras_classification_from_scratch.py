# This code is adapted from the Keras documentation and can serve as a baseline   
# https://keras.io/examples/nlp/text_classification_from_scratch/#text-classification-from-scratch

# import Packages
import keras 
import tensorflow as tf

# Load the dataset and split it into training, validation and test set
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/Labeled/Train/",
                                                          labels="inferred",
                                                          label_mode="categorical",
                                                          batch_size=64,
                                                          validation_split=0.2,
                                                          subset="training",
                                                          seed=1337)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/Labeled/Train/",
                                                          labels="inferred",
                                                          label_mode="categorical",
                                                          batch_size=64,
                                                          validation_split=0.2,
                                                          subset="validation",
                                                          seed=1337)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(directory="/home/jupyter-ozkan_ma/Labeled/Test/",
                                                         labels="inferred",
                                                         label_mode="categorical",
                                                         batch_size=64)     

# Import a TextVectorization layer to transform text into numbers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

vectorize_layer = TextVectorization(max_tokens=20000,
                                    output_mode="int",
                                    output_sequence_length=500)

# Let's make a text-only dataset (no labels):
text_ds = raw_train_ds.map(lambda x, y: x)
# Let's call `adapt`:
vectorize_layer.adapt(text_ds)                                   


# Function to vectorize the text
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

# Model constants.
max_features = 20000
embedding_dim = 128

# Building the model, since layers 1-5 can be reused a different naming schema is followed
# In other models only layers from six onwards will be different 

from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
layer_1 = layers.Embedding(max_features, embedding_dim)(inputs)
layer_2 = layers.Dropout(0.5)(layer_1)

# Conv1D + global max pooling
layer_3 = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(layer_2)
layer_4 = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(layer_3)
layer_5 = layers.GlobalMaxPooling1D()(layer_4)

# We add a vanilla hidden layer:
layer_6 = layers.Dense(128, activation="relu")(layer_5)
layer_7 = layers.Dropout(0.5)(layer_6)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(5, activation="softmax", name="predictions")(layer_7)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 3
# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Evaluate the model on the test set
model.evaluate(test_ds)