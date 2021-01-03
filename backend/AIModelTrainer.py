from multiprocessing import Process
import django
django.setup()
from .AIModel import AIModel
from .playlist import Playlist
import backend.UserTracker as UserTracker
from .models import PlaylistModel, AIConfig, AIEncoding, User
import sys
import pandas as pd
from .spotify import get_user_playlists, pull_playlist
from numpy.random import seed
import numpy as np
from math import pi
import pandas as pd
from keras.layers import Embedding, concatenate, Dense, Input, Flatten
from keras import Model
from keras import optimizers
import tensorflow as tf
import math

# Getting rid of the stochasticy from our models by fixing the random number generator
seed(1)
tf.compat.v1.disable_eager_execution()

class AIModelTrainer(AIModel, Process):
    def __init__(self, uid):
       Process.__init__(self)
       AIModel.__init__(self, uid)

    def run(self):
        self.create_model()
        user_tracker_process = UserTracker.UserTracker(self.uid)
        user_tracker_process.start()
        sys.exit(0)
    
    def create_model(self):
        user_data = self.gather_user_data()
        X_audio, X_artist, Y = self.split(user_data, training=True)
        user_data.to_csv('user_data_test.csv', index=False, encoding='utf-8')
        del user_data
        X_audio = self.scale_continuous(X_audio)
        X_artist = self.encode_artist(X_artist)
        X_artist = self.add_padding(X_artist)
        self.train_encoder(Y)
        self.save_encodings()
        hot_Y = self.label_transform(Y)

        pd.DataFrame(columns=self.audio_features, data=X_audio).to_csv(
            'x_audio.csv', index=False, encoding='utf-8')
        X_artist.to_csv('x_artist.csv', index=False, encoding='utf-8')

        X_artist = self.tf_transform(X_artist)
        self.setup(X_artist, hot_Y)
        self.train(X_audio, X_artist, hot_Y)

        self.save()

    def setup(self, X_artist, hot_Y):
        # I add 1 because i want the model to work when we have new artists thrown at it
        artist_embd_dim = math.ceil(np.log2(len(np.unique(X_artist)))+1)
        # --- Model Architecture ---
        n_numerical_feats = 15
        # number of classes/playlists
        n_classes = hot_Y.shape[1]
        print(f'n_classes: {n_classes}')
        # artist input to embeddings
        artist_input = Input(shape=(self.maxlen,), name='artist_input')
        artist_embedding = Embedding(
            self.vocab_size, artist_embd_dim, input_length=self.maxlen)(artist_input)
        artist_vec = Flatten()(artist_embedding)
        # numerical features input
        numerical_input = Input(
            shape=(n_numerical_feats), name='numeric_input')

        # input layer
        merged = concatenate([numerical_input, artist_vec])

        # hidden layers
        # we want to make the network abstract the input information by reducing the dimensions
        size_input = n_numerical_feats+(artist_embd_dim*self.maxlen)
        size_hidden1 = int(size_input*32)
        size_hidden2 = int(size_input*32)

        hidden1 = Dense(size_hidden1, activation='relu')(merged)
        hidden2 = Dense(size_hidden2, activation='relu')(hidden1)

        # output layers
        output = Dense(n_classes, activation='softmax')(hidden2)

        # define the model
        model = Model([numerical_input, artist_input], output)
        # compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'], experimental_run_tf_function=False)

        self.model = model

    def train(self, X_audio, X_artist, Y):
        # --- Model configuration ---
        batch_size = 8
        epochs = 10
        # Fit data to model
        # print(f'X_audio[0]: {X_audio[0]}, shape: {X_audio.shape}')
        # print(f'X_artist[0]: {X_artist}, shape: {X_artist.shape}')
        # print(f'Y: {Y}, shape: {Y.shape}')

        self.model.fit([X_audio, X_artist], [Y],
                       batch_size=batch_size, epochs=epochs, verbose=2)

    def save(self):
        self.model.save(f'models/{self.uid}.h5')

        model = AIConfig(uid=self.uid, maxlen=self.maxlen,
                         vocab_size=self.vocab_size)
        model.save()
        # p_user_songs_tracker = Process(target=user_songs_tracker, args=(user, maxlen, vocab_size, encoder))
        # p_user_songs_tracker.start()
    
    def gather_user_data(self):
        user = User.objects.filter(uid=self.uid)[0]
        playlists_json = get_user_playlists(user)
        user_data = pd.DataFrame()
        for item in playlists_json['items']:
            # We only want the user created playlists
            if item['owner']['id'] == self.uid:
                playlist = Playlist(id=item['id'], name=item['name'], user=user)
                pull_playlist(user, playlist)
                playlist.add_audio_features()
                playlist.to_df()
                playlist.save()
                user_data = user_data.append(playlist.df)
                del playlist
        user_data.dropna(inplace=True)
        return user_data
        
def get_maxlen(X_artist):
    # Find max len to do padding
    maxlen = -1
    for idx, row in X_artist.iterrows():
        if len(row['artist']) > maxlen:
            maxlen = len(row['artist'])
    # I do this so future examples with longer artist names can still fit into the model
    return maxlen + 10