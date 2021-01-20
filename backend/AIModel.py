import sys
import multiprocessing
import django
django.setup()
from .playlist import Playlist
from .models import PlaylistModel, AIConfig, AIEncoding, User

from keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

class AIModel():
    def __init__(self, uid):
        self.uid = uid
        self.maxlen = None
        self.vocab_size = None
        self.encoder = LabelEncoder()
        self.audio_features = ['len_ms', 'explicit',
                               'popularity', 'key', 'mode', 'time_signature',
                               'acousticness', 'danceability', 'energy',
                               'instrumentalness', 'liveness', 'loudness',
                               'speechiness', 'valence', 'tempo']
        self.model = None


    def load_user_model(self):
        config = AIConfig.objects.filter(uid=self.uid)[0]
        self.maxlen = config.maxlen
        self.vocab_size = config.vocab_size
        self.model = load_model(f'models/{self.uid}.h5')
        self.load_encodings()
    
    def predict(self, data):
        X_audio, X_artist, _ = self.split(data, training=False)
        # data.to_csv('predict_data_test.csv', index=False, encoding='utf-8')
        del data
        X_audio = self.scale_continuous(X_audio)
        X_artist = self.encode_artist(X_artist)
        X_artist = self.add_padding(X_artist)

        # pd.DataFrame(columns=self.audio_features, data=X_audio).to_csv(
        #     'predict_x_audio.csv', index=False, encoding='utf-8')
        # X_artist.to_csv('predict_x_artist.csv', index=False, encoding='utf-8')

        X_artist = self.tf_transform(X_artist)
        predictions = self.model.predict([X_audio, X_artist])
        playlist_id = self.encoder.inverse_transform(predictions.argmax(axis=-1))[0]
        return playlist_id
        

    def scale_continuous(self, X_audio):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_audio = scaler.fit_transform(X_audio)

        return X_audio

    def encode_artist(self, X_artist):
        # Integer encode artist names
        # We will estimate the vocabulary size of (unique_artists*1000), which is much larger than needed to
        # reduce the probability of collisions from the hash function.
        if self.vocab_size is None:
            self.vocab_size = len(X_artist['artist'].unique()) * 1000

        for idx, row in X_artist.iterrows():
            X_artist.at[idx, 'artist'] = one_hot(
                row['artist'], self.vocab_size)

        return X_artist

    def add_padding(self, X_artist):
        if self.maxlen is None:
            self.maxlen = get_maxlen(X_artist)
        # Perform padding
        encodings_padded = []
        for idx, row in X_artist.iterrows():
            encodings_padded.append(row['artist'])

        encodings_padded = pad_sequences(
            encodings_padded, maxlen=self.maxlen, padding='post')

        for (idx, row), encoding in zip(X_artist.iterrows(), encodings_padded):
            X_artist.at[idx, 'artist'] = encoding
        
        return X_artist

    def label_transform(self, Y):
        # Convert target playlist to one hot encoded playlist for Neural Network
        # First encode target values as integers from string
        hot_Y = self.encoder.transform(Y)
        # Then perform one hot encoding
        hot_Y = np_utils.to_categorical(hot_Y)
        return hot_Y

    def train_encoder(self, Y):
        self.encoder.fit(Y)

    def save_encodings(self):
        encodings = {}
        for i in list(self.encoder.classes_):
            encodings[i] = self.encoder.transform([i])[0]

        playlists = PlaylistModel.objects.filter(uid=self.uid)
        for playlist in playlists:
            encoding = AIEncoding(uid=self.uid, spotify_id=playlist.spotify_id,
                                  playlist_name=playlist.name, encoding=encodings[playlist.spotify_id])
            encoding.save()

    def load_encodings(self):
        encodings = AIEncoding.objects.filter(uid=self.uid)
        classes = []
        for encoding in encodings:
            classes.append(encoding.spotify_id)

        self.encoder.classes_ = np.array(classes, dtype=object)

    def split(self, data, training):
        if training:
            X = data.drop(columns='playlist_id')
            Y = data['playlist_id']
        else:
            X = data
            Y = None

        X_artist, X_audio = pd.DataFrame(
            columns=['artist']), pd.DataFrame(columns=self.audio_features)

        # Split data into audio and categorical data
        X_artist['artist'] = X['artist'].values
        X_audio = X.drop(columns=['artist'])

        return X_audio, X_artist, Y

    def tf_transform(self, X_artist):
        # Convert data to friendly arrays of TensorFlow
        X_artist = X_artist['artist'].values.tolist()
        X_artist = np.asarray(X_artist).astype('float32')

        return X_artist


def get_maxlen(X_artist):
    # Find max len to do padding
    maxlen = -1
    for idx, row in X_artist.iterrows():
        if len(row['artist']) > maxlen:
            maxlen = len(row['artist'])
    # I do this so future examples with longer artist names can still fit into the model
    return maxlen + 10
