import config
from playlist import Playlist
import pandas as pd

# TODO: Crear pandas.DataFrame a partir de objeto playlist

playlists_map = {
    'Trapeo': '37i9dQZF1DWXCGnD7W6WDX',
    '100% Cumbia': '37i9dQZF1DX8yLfjPY8emY'
}

playlists_list = []

config.init_globals()

for name, id in playlists_map.items():
    playlist = Playlist(id=id, name=name, fields=config.standard_fields)
    config.spoti.pull_playlist(playlist)
    playlist.add_audio_features()
    playlists_list.append(playlist)

for playlist in playlists_list:
    playlist.to_csv()
