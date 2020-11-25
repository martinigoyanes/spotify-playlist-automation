import config
from playlist import Playlist
import pandas as pd
import csv
import os

# TODO: Crear pandas.DataFrame a partir de objeto playlist
# TODO: Be able to add audio features to more than 100 songs


playlists_map = {
    'Bonfire Songs':'6Oh477v4CTzLTBfpbUK8Nf',
    'KPop 2020 Top hits':'4UTvZBu04lpLMyPcTVo2dt',
    '90S ROCK  ANTHEMS':'37i9dQZF1DX1rVvRgjX59F',
    'TOMORROWLAND 2020 PLAYLIST':'3VXReCeetN58c1clj9u8ZK',
    'LEGIT':'6XKlfK2tVncAqbVLDFruQJ',
    'NIGRO':'5RD8Qp8XGHC4uemGFJ9Qep',
    'REGGAETON':'62J5t9At60EiRLwGf6XTsT',
    'CHILL':'62EDgX30ollO8Dhtt7kUaT',
    'ESPAÑOLA':'7lPdwGVpFcfMsbvNQ8Nmpj',
    'ROMANTICISM':'2dKJabpNOGINjaHPmd84Sr',
    'ESPAÑOLAS (ESTOPA, AMARAL, FITO, ETC) RO':'0jAoxn59h4uk2iYm6P2lXT',
    'JAZZ & CIGGARETTES >> COOL JAZZ':'5mjpDgOuiDYXUf6zTKOkLm',
    'FOLK INDIE, ETC':'4zCu70NqiL0W0nJvNEu0Io',
    'SLAP':'6rAB7xc2gTxMeLamm1zRlI',
    'PROGRESSIVE (PINK FLOYD)':'5HaSpkNg4ndgsgWyIO8wSP',
    'OLD GRAMOPHONE(1920,1930,1940)':'2UOUjGXDLYGnRYOokC37c1',
    'ANDIAAAAMO':'3LzHT0tgC3DEmL6zRgAUIv',
    'NICE':'5NPXtm9qaqLYTnCNMPaa7r',
    'CLASSICSZ MEDLEY CLASSIC ROCK 60,70,80':'4Fh0fZZ8YjLKT8f4GRWbII',
    'EN EL TRABAJO':'37i9dQZF1DX62RIcEOaBWi',
    'DEEP FOCUS':'37i9dQZF1DWZeKCadgRdKQ',
    'LOFI BEATS':'37i9dQZF1DWWQRwui0ExPn',
    'NATURE SOUNDS':'37i9dQZF1DX4PP3DA4J0N8',
    'FLAMENCO':'37i9dQZF1DWTfrrcwZAlVH',
    'YOGA':'37i9dQZF1DX9uKNf5jGX6m',
    'POWER HOUR':'37i9dQZF1DX32NsLKyzScr',
    'ALL OUT 00S':'37i9dQZF1DX4o1oenSJRJd',
    'CANTAUTORES ESPAÑOLES (DE AYER, HOY Y SIEMPRE)':'5nLAjT7pQGoh2gfN1EHRDZ',
    'MARIACHIS CLASSICS':'4dfGUZZed5A0dws5x2dZ1V',
    'VERANO 2020':'74yelSrmOWJdnviXV30QR5',
    'JAZZ PERCUSION':'6xEIPEalTLbReA16UYNo3m',
    'THE FEELS CHICO':'3HG9hQZhn2p1pbEMO9iyCf',
    'COCINA CON CUMBIA':'37i9dQZF1DX3spgx4EgoDM',
    'LONG STAR BLUES':'37i9dQZF1DXeaMD6NdSui3'
}

playlists_list = []

config.init_globals()

if not os.path.exists('datasets'):
    os.makedirs('datasets')

try:
    with open('datasets/jaime-playlists.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Name', 'Artists', 'Album', 'Duration(ms)', 'Explicit', 'Popularity', 'Key', 'Mode', 'Time Signature', 'Acousticness',
                         'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Playlist'])
                         
        for name, id in playlists_map.items():
            playlist = Playlist(id=id, name=name, fields=config.standard_fields)
            config.spoti.pull_playlist(playlist)
            playlist.add_audio_features()
            # playlist.to_csv()
            
            for song in playlist.song_list:
                writer.writerow([song.name, ",".join(song.artists), song.album, song.len_ms, song.explicit, song.popularity,
                                    song.key, song.mode, song.time_signature, song.acousticness, song.danceability,
                                    song.energy, song.instrumentalness, song.liveness, song.loudness, song.speechiness,
                                    song.valence, song.tempo, playlist.name])
except BaseException as e:
    print('BaseException:', filename, e)
