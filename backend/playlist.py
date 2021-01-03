from .song import Song 
from .spotify import get_audio_features_several
from .models import PlaylistModel
import math
from decimal import Decimal, getcontext
import csv
import os
import pandas as pd
class Playlist():
    def __init__(self, id, name, user):
        self.id = id
        self.name = name
        self.user = user
        self.fields = 'items(track(id, album(name), artists(name), explicit, name, duration_ms, popularity))'
        self.song_list = []
        columns=(   'artist', 'len_ms', 'explicit', 
                    'popularity', 'key', 'mode', 'time_signature',
                    'acousticness', 'danceability', 'energy',
                    'instrumentalness', 'liveness', 'loudness',
                    'speechiness', 'valence', 'tempo', 'playlist_id')
        self.df = pd.DataFrame(columns=columns)

    def json_to_playlist(self, playlist_json):

        usefull_tracks = (
            item for item in playlist_json['items'] if item['track'] is not None)
        
        usefull_songs = (
            song for song in usefull_tracks if song['track']['id'] is not None)

        for item in usefull_songs:
            song_name = item['track']['name']
            song_artist = item['track']['artists'][0]['name']
            song_album = item['track']['album']['name']
            song_len_ms = item['track']['duration_ms']
            song_id = item['track']['id']
            song_explicit = item['track']['explicit']
            song_popularity = item['track']['popularity']
            song = Song(name=song_name, artist=song_artist, album=song_album,
                             len_ms=song_len_ms, explicit=song_explicit,
                             popularity=song_popularity, id=song_id)
            self.song_list.append(song)

    def add_audio_features(self):
        ids = [song.id for song in self.song_list]
        # We can only request 100 ids at a time, so if there are more we need to talk to the api more times
        # 3435 songs -> 34.35 requests -> 34 requests of 100 songs, 1 of 35(0.35*1000)
        getcontext().prec = 2
        n_requests = len(ids)/100  # 34.35 requests
        n_requests_low = math.floor(n_requests)  # 34 requests of 100 songs
        # (34.35 - 34)*100 -> 1 of 35 songs
        partial_request_len = int(
            float(Decimal(n_requests) - Decimal(n_requests_low))*100)

        # Full requests of 100
        if n_requests >= 1:
            for n in range(1, n_requests_low + 1):
                audio_features_json = get_audio_features_several(
                    ids[((n-1)*100):(n*100)], self.user)
                for audio_features, song in zip(audio_features_json['audio_features'], self.song_list[((n-1)*100):(n*100)]):
                    song.add_audio_features(audio_features)
                # Partial request of < 100 (last request)
                if n == (n_requests_low + 1) and partial_request_len > 0:
                    audio_features_json = get_audio_features_several(
                        ids[(n*100):((n*100)+partial_request_len)], self.user)
                    for audio_features, song in zip(audio_features_json['audio_features'], self.song_list):
                        song.add_audio_features(audio_features)
        else:
            # Partial request of < 100
            audio_features_json = get_audio_features_several(
                ids[0:partial_request_len], self.user)
            for audio_features, song in zip(audio_features_json['audio_features'], self.song_list):
                song.add_audio_features(audio_features)
    
    def to_df(self):
        for song in self.song_list:
            s = {}
            s['artist'] = song.artist, 
            s['len_ms'] = song.len_ms, 
            s['explicit']  = song.explicit,
            s['popularity']  = song.popularity,
            s['key']  = song.key,
            s['mode']  = song.mode,
            s['time_signature']  = song.time_signature,
            s['acousticness']  = song.acousticness,
            s['danceability'] = song.danceability,
            s['energy'] =song.energy,
            s['instrumentalness'] = song.instrumentalness,
            s['liveness'] = song.liveness,
            s['loudness'] = song.loudness,
            s['speechiness'] = song.speechiness,
            s['valence'] = song.valence,
            s['tempo'] = song.tempo,
            s['playlist_id'] = self.id,
            self.df = self.df.append(pd.DataFrame(s))
            del song
        del self.song_list
    
    def save(self):
        playlist = PlaylistModel(uid=self.user.uid,spotify_id=self.id,name=self.name)
        playlist.save()