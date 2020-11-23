import time
import glob
import os

from mutagen.id3 import ID3, APIC
from mutagen.mp3 import EasyMP3
from urllib.request import urlopen

import config


class Song:
    artists = []
    key = None
    mode = None
    time_signature = None
    acousticness = None
    danceability = None
    energy = None
    instrumentalness = None
    liveness = None
    loudness = None
    speechiness = None
    valence = None
    tempo = None

    def __init__(self, name, artists, album, len_ms, id, explicit, popularity):
        self.name = name
        self.artists = artists
        self.fullname = name
        for artist in artists:
            self.fullname += f' - {artist}'
        self.album = album
        self.len_ms = len_ms
        self.id = id
        self.explicit = explicit
        self.popularity = popularity

    def add_audio_features(self):
        audio_features_json = config.spoti.get_audio_features(self.id)
        self.key = audio_features_json['key']
        self.mode = audio_features_json['mode']
        self.time_signature = audio_features_json['time_signature']
        self.acousticness = audio_features_json['acousticness']
        self.danceability = audio_features_json['danceability']
        self.energy = audio_features_json['energy']
        self.instrumentalness = audio_features_json['instrumentalness']
        self.liveness = audio_features_json['liveness']
        self.loudness = audio_features_json['loudness']
        self.speechiness = audio_features_json['speechiness']
        self.valence = audio_features_json['valence']
        self.tempo = audio_features_json['tempo']
