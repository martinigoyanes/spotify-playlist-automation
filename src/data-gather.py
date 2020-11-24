import config
from playlist import Playlist
import pandas as pd

# TODO: Crear pandas.DataFrame a partir de objeto playlist
# TODO: Be able to add audio features to more than 100 songs

playlists_map = {
    'Trapeo': '37i9dQZF1DWXCGnD7W6WDX',
    '100% Cumbia': '37i9dQZF1DX8yLfjPY8emY',
    'A Perriar': '3BKDg8HYzaxME5JDY25osX'
}

playlists_list = []

config.init_globals()

for name, id in playlists_map.items():
    playlist = Playlist(id=id, name=name, fields=config.standard_fields)
    config.spoti.pull_playlist(playlist)
    playlist.add_audio_features()
    playlist.to_csv()
