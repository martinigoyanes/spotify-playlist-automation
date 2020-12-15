import tensorflow as tf
from keras import optimizers
from keras import Model
from keras.layers import Embedding, concatenate, Dense, Input, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Getting rid of the stochasticy from our models by fixing the random number generator
from numpy.random import seed
seed(1)


def plot_training(history, embedding_dim, batch_size):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(
        f'Model Accuracy embedding_dims={embedding_dim}, batch_size={batch_size}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(
        f'Model Loss embedding_dims={embedding_dim}, batch_size={batch_size}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


df = pd.read_csv("datasets/clean-playlists-one-artist.csv")

# --- Use playlists chose in Data Analysis section ---
chosen_pls = ['Tuff', 'Yoga', 'PunkEspanol',
              'RapEspanol(TLob)', 'Metal', 'Janngueo', 'DubReggae', 'DailyMix3', 'DailyMix5', 'CountryNights']
df_10_playlist = df.loc[df['Playlist'].isin(chosen_pls)]

# Scale continuous data
features = ['Popularity', 'Acousticness', 'Danceability',
            'Instrumentalness', 'Loudness', 'Encoded Encoded Artist']
scaler = MinMaxScaler(feature_range=(0, 1))
X_continuous = scaler.fit_transform(df_10_playlist[features[:5]])
X_continuous = pd.DataFrame(X_continuous)

# --- One Hot Encode Classes ---
# Convert target playlist to one hot encoded playlist for Neural Network
# First encode target values as integers from string
Y = df_10_playlist['Playlist']
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
# Then perform one hot encoding
Y = np_utils.to_categorical(Y)

# --- Encode Artist Names ---
# Integer encode artist names
X = X_continuous
X['Artist'] = df_10_playlist['Artist'].values
# We will estimate the vocabulary size of (unique_artists*1000), which is much larger than needed to
# reduce the probability of collisions from the hash function. (unique_artists = 510)
vocab_size = len(df['Artist'].unique()) * 1000
X['Encoded Artist'] = ''
for idx, row in X.iterrows():
    X.at[idx, 'Encoded Artist'] = one_hot(row['Artist'], vocab_size)
    #print("The encoding for ", row['Artist'] ,"is :", X.at[idx, 'Encoded Artist'])

# Padding
# Find max len to do padding
maxlen = -1
for idx, row in X.iterrows():
    if len(row['Encoded Artist']) > maxlen:
        maxlen = len(row['Encoded Artist'])

# Perform padding
encodings_padded = []
for idx, row in X.iterrows():
    encodings_padded.append(row['Encoded Artist'])

encodings_padded = pad_sequences(
    encodings_padded, maxlen=maxlen, padding='post')

X.drop(columns='Encoded Artist')
X['Encoded Artist'] = None
for (idx, row), encoding in zip(X.iterrows(), encodings_padded):
    X.at[idx, 'Encoded Artist'] = encoding
X.drop(columns=['Artist'], inplace=True)


# --- Split data into test and train data ---
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# Split data into continuous and categorical data
x_train_artist, x_test_artist = x_train['Encoded Artist'], x_test['Encoded Artist']
x_train_continuous, x_test_continuous = x_train.drop(
    columns=['Encoded Artist']), x_test.drop(columns=['Encoded Artist'])

# Convert data to friendly arrays of TensorFlow
x_train_artist, x_test_artist = x_train_artist.values, x_test_artist.values
x_train_artist, x_test_artist = x_train_artist.tolist(), x_test_artist.tolist()
x_train_artist, x_test_artist = np.asarray(x_train_artist).astype(
    'float32'), np.asarray(x_test_artist).astype('float32')

x_train_continuous, x_test_continuous = x_train_continuous.values, x_test_continuous.values

#! Model
tf.compat.v1.disable_eager_execution()

n_numerical_feats = 5
n_classes = 10  # number of classes/playlists

results = {}
# Model configuration
batch_size = [2, 8, 16, 32]
epochs = [8, 10, 15, 20, 25]
# embedding dimension for each categorical feature = log2(vocab)
artist_embd_dim = [9, 18]

for b in batch_size:
    results[f'batch_size:{b}'] = {}
    for e in epochs:
        results[f'batch_size:{b}'][f'epochs:{e}'] = {}
        for dim in artist_embd_dim:
            # artist embeddings
            artist_input = Input(shape=(maxlen,), name='artist_input')
            artist_embedding = Embedding(
                vocab_size, dim, input_length=maxlen)(artist_input)
            artist_vec = Flatten()(artist_embedding)
            # numerical features
            numerical_input = Input(
                shape=(n_numerical_feats), name='numeric_input')

            # input layer
            merged = concatenate([numerical_input, artist_vec])

            # hidden layers
            size_input = n_numerical_feats+(dim*6)
            # we want to make the network abstract the input information by increasing the dimensions
            size_hidden1 = int(size_input*32)

            hidden1 = Dense(size_hidden1, activation='relu')(merged)
            output = Dense(n_classes, activation='softmax')(hidden1)
            # define the model
            model = Model([numerical_input, artist_input], output)

            # compile the model
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[
                          'accuracy'], experimental_run_tf_function=False)

            # summarize the model
            # print(model.summary())

            # train model
            history = model.fit(x=[x_train_continuous, x_train_artist], y=[y_train],
                                batch_size=b, epochs=e, verbose=2, validation_split=0.10)
            plot_training(history, dim, b)
            loss, accuracy = model.evaluate(x=[x_test_continuous, x_test_artist], y=[y_test],
                                            batch_size=b, verbose=2)

            results[f'batch_size:{b}'][f'epochs:{e}'][f'embed_dim:{dim}'] = (
                loss, accuracy)
            print(results)
print(results)
