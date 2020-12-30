from .models import User, SongModel
from django.utils import timezone
from datetime import timedelta
from .credentials import CLIENT_ID, CLIENT_SECRET
from requests import post, put, get

from .spotify import get_user_playlists, pull_playlist, get_user_tracks_json, get_audio_features, put_song_in_playlist
from .playlist import Playlist

import math
import tensorflow as tf
from keras import optimizers
from keras import Model
from keras.layers import Embedding, concatenate, Dense, Input, Flatten
from keras.models import load_model
import pandas as pd
from math import pi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from django.core.files import File
from django.core.files.base import ContentFile
# Getting rid of the stochasticy from our models by fixing the random number generator
from numpy.random import seed
seed(1)
tf.compat.v1.disable_eager_execution()


def get_user(uid=None, session_key=None):
    user_tokens = User.objects.filter(
        uid=uid) if uid else User.objects.filter(curr_session=session_key)
    if user_tokens.exists():
        return user_tokens[0]
    else:
        return None


def update_or_create_user(curr_session, access_token, token_type, expires_in, refresh_token):

    response = get('https://api.spotify.com/v1/me', headers={
        'Authorization': f'Bearer {access_token}'
    }).json()
    uid = response.get('id')

    user = get_user(uid=uid)
    expires_in = timezone.now() + timedelta(seconds=expires_in)

    if user:
        user.access_token = access_token
        user.refresh_token = refresh_token
        user.expires_in = expires_in
        user.token_type = token_type
        user.curr_session = curr_session
        user.save(update_fields=['access_token',
                                 'refresh_token', 'expires_in', 'token_type', 'curr_session'])
    else:
        create_user(curr_session, uid, access_token,
                    refresh_token, token_type, expires_in)
        # tokens = User(uid=uid, access_token=access_token,
        #               refresh_token=refresh_token, token_type=token_type, expires_in=expires_in)
        # tokens.save()


def create_user(curr_session, uid, access_token, refresh_token, token_type, expires_in):
    user = User(uid=uid, access_token=access_token,
                refresh_token=refresh_token,
                token_type=token_type, expires_in=expires_in, curr_session=curr_session)
    user.save()

    gather_user_data(user)
    maxlen, vocab_size, encoder = train_model(user)
    predict_playlist_from_song(user, maxlen, vocab_size, encoder)
    
def gather_user_data(user):
    playlists_json = get_user_playlists(user)
    for item in playlists_json['items']:
        # We only want the user created playlists
        if item['owner']['id'] == user.uid:
            playlist = Playlist(id=item['id'], name=item['name'], user=user,
                                fields='items(track(id, album(name), artists(name), explicit, name, duration_ms, popularity))')
            pull_playlist(user, playlist)
            playlist.add_audio_features()
            playlist.save()

def train_model(user):
    # songs into DataFrame
    user_data = SongModel.objects.filter(uid=user.uid).values()
    df = pd.DataFrame.from_records(user_data)
    df.dropna(inplace=True)
    df.drop(columns=['id', 'playlist_id', 'uid'], inplace=True)
    df.to_csv("user_data_test.csv", index=False, encoding='utf-8')
    df = pd.read_csv("user_data_test.csv")
    # Scale continuous data
    # we want now all continuous features
    non_wanted_features = ['name', 'artist', 'playlist_name', 'album']

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_continuous = df.drop(columns=non_wanted_features)
    features = X_continuous.columns

    X_continuous = scaler.fit_transform(X_continuous)
    X_continuous = pd.DataFrame(X_continuous)
    X_continuous.columns = features

    X_continuous

    # --- Encode Artist Names ---
    # Integer encode artist names
    X = X_continuous
    X['Artist'] = df['artist'].values
    # We will estimate the vocabulary size of (unique_artists*1000), which is much larger than needed to
    # reduce the probability of collisions from the hash function. (unique_artists = 510)
    vocab_size = len(df['artist'].unique()) * 1000
    X['Encoded Artist'] = ''
    for idx, row in X.iterrows():
        X.at[idx, 'Encoded Artist'] = one_hot(row['Artist'], vocab_size)

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

    # --- One Hot Encode Classes ---
    # Convert target playlist to one hot encoded playlist for Neural Network
    # First encode target values as integers from string
    Y = df['playlist_name']
    encoder = LabelEncoder()
    encoder.fit(Y)
    hot_Y = encoder.transform(Y)
    # Then perform one hot encoding
    hot_Y = np_utils.to_categorical(hot_Y)

    # --- Split data into test and train data ---
    x_train, x_test, hot_y_train, hot_y_test = train_test_split(
        X, hot_Y, test_size=0.3)

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

    # --- Model configuration ---
    batch_size = 8
    epochs = 10
    artist_embd_dim = math.ceil(np.log2(len(df['artist'].unique())))

    # --- Model Architecture ---
    n_numerical_feats = 15
    # number of classes/playlists
    n_classes = len(df['playlist_name'].unique())
    # artist input to embeddings
    artist_input = Input(shape=(maxlen,), name='artist_input')
    artist_embedding = Embedding(
        vocab_size, artist_embd_dim, input_length=maxlen)(artist_input)
    artist_vec = Flatten()(artist_embedding)
    # numerical features input
    numerical_input = Input(shape=(n_numerical_feats), name='numeric_input')

    # input layer
    merged = concatenate([numerical_input, artist_vec])

    # hidden layers
    # we want to make the network abstract the input information by reducing the dimensions
    size_input = n_numerical_feats+(artist_embd_dim*maxlen)
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

    # Fit data to model
    model.fit([x_train_continuous, x_train_artist], [hot_y_train], batch_size=batch_size, epochs=epochs, verbose=2,
              validation_data=([x_test_continuous, x_test_artist], [hot_y_test]))

    model.save(f'{user.uid}.h5')
    
    return maxlen, vocab_size, encoder

def predict_playlist_from_song(user, maxlen, vocab_size, encoder):
    # Get user last liked song
    tracks_json = get_user_tracks_json(user, limit=20)
    item = tracks_json['items'][0]
    audio_features_json = get_audio_features(item['track']['id'], user)
    song = {
        'name': item['track']['name'],
        'artist': item['track']['artists'][0]['name'],
        'album': item['track']['album']['name'],
        'len_ms': item['track']['duration_ms'],
        'explicit': item['track']['explicit'],
        'popularity': item['track']['popularity'],
        'key': audio_features_json['key'],
        'mode': audio_features_json['mode'],
        'time_signature': audio_features_json['time_signature'],
        'acousticness': audio_features_json['acousticness'],
        'danceability': audio_features_json['danceability'],
        'energy': audio_features_json['energy'],
        'instrumentalness': audio_features_json['instrumentalness'],
        'liveness': audio_features_json['liveness'],
        'loudness': audio_features_json['loudness'],
        'speechiness': audio_features_json['speechiness'],
        'valence': audio_features_json['valence'],
        'tempo': audio_features_json['tempo']
    }
    df = pd.DataFrame(song, index=[0])
    df.to_csv("user_song_test.csv", index=False, encoding='utf-8')

    # Scale continuous data
    # we want now all continuous features
    non_wanted_features = ['name', 'artist', 'album']

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_continuous = df.drop(columns=non_wanted_features)
    features = X_continuous.columns

    X_continuous = scaler.fit_transform(X_continuous)
    X_continuous = pd.DataFrame(X_continuous)
    X_continuous.columns = features

    # --- Encode Artist Names ---
    # Integer encode artist names
    X = X_continuous
    X['Artist'] = df['artist'].values
    X['Encoded Artist'] = ''
    for idx, row in X.iterrows():
        X.at[idx, 'Encoded Artist'] = one_hot(row['Artist'], vocab_size)

    # Padding

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

    # Convert data to friendly arrays of TensorFlow
    X_artist = X['Encoded Artist']
    X_continuous = X.drop(columns=['Encoded Artist'])

    X_artist = X_artist.values
    X_artist = X_artist.tolist()
    X_artist = np.asarray(X_artist).astype('float32')

    X_continuous = X_continuous.values

    loaded_model = load_model(f'{user.uid}.h5')
    predictions = loaded_model.predict([X_continuous, X_artist])
    playlist = encoder.inverse_transform(predictions.argmax(axis=-1))[0]

    playlist_id = SongModel.objects.filter(
        playlist_name=playlist).values()[0]['playlist_id']
    song_uri = item['track']['uri']
    resp = put_song_in_playlist(user, playlist_id, [song_uri])


def is_spotify_authenticated(session_key):
    user = get_user(session_key=session_key)
    if user:
        expiry = user.expires_in
        if expiry <= timezone.now():
            refresh_spotify_token(user)

        return True

    return False


def refresh_spotify_token(user):
    refresh_token = user.refresh_token

    response = post('https://accounts.spotify.com/api/token', data={
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }).json()

    access_token = response.get('access_token')
    token_type = response.get('token_type')
    expires_in = response.get('expires_in')
    refresh_token = response.get('refresh_token')

    update_or_create_user(
        user.curr_session, access_token, token_type, expires_in, refresh_token)
