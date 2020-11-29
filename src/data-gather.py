import config
from playlist import Playlist
import pandas as pd
import csv
import os

# TODO: Crear pandas.DataFrame a partir de objeto playlist

filename = 'datasets/playlists'
playlists = {}
with open(f'{filename}.txt') as f:
    for line in f:
       (key, val) = line.split()
       playlists[key] = val


config.init_globals()

# try:
with open(f'{filename}.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Artists', 'Album', 'Duration(ms)', 'Explicit', 'Popularity', 'Key', 'Mode', 'Time Signature', 'Acousticness',
                        'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Playlist'])

    for name, id in playlists.items():
        playlist = Playlist(id=id, name=name, fields=config.standard_fields)
        config.spoti.pull_playlist(playlist)
        playlist.add_audio_features()
        playlist.to_csv()

        for song in playlist.song_list:
            writer.writerow([song.name, ",".join(song.artists), song.album, song.len_ms, song.explicit, song.popularity,
                                song.key, song.mode, song.time_signature, song.acousticness, song.danceability,
                                song.energy, song.instrumentalness, song.liveness, song.loudness, song.speechiness,
                                song.valence, song.tempo, playlist.name])
# except BaseException as e:
#     print('BaseException:', f'{filename}.csv', e)
