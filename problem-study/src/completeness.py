import config
from   playlist import Playlist
import pandas as pd
import csv
import os

# config.init_globals()
## From .csv without album, explicit and popularity create new csv with them
# names = ['dinner_tracks','party_tracks','sleep_tracks', 'workout_tracks']
# for name in names:
#     df = pd.read_csv(f'datasets/kaggle/{name}.csv')
#     playlist = Playlist(id='', name=name, fields='')
#     # Get ids from all the songs
#     ids = [idx for idx in df['id'].values] 
#     playlist.load_tracks(ids)
#     playlist.add_audio_features()
#     playlist.to_csv()

## Append playlist column
# names = ['Dinner Tracks','Party Tracks','Sleep Tracks', 'Workout Tracks']
# for name in names:
#     df = pd.read_csv(f'datasets/playlists/{name}.csv')
#     df['Playlist'] = name
#     df.to_csv(f'datasets/playlists/{name}.csv')

## Combine all playlists
# names = ['Dinner Tracks','Party Tracks','Sleep Tracks', 'Workout Tracks']
# combined_csv = pd.concat([pd.read_csv(f'datasets/playlists/{name}.csv') for name in names ])
# combined_csv.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'], inplace=True)
# combined_csv.to_csv( "datasets/kaggle playlists.csv", index=False, encoding='utf-8')

## Combine wth our playlists
names = ['playlists','kaggle playlists']
combined_csv = pd.concat([pd.read_csv(f'datasets/{name}.csv') for name in names ])
combined_csv.to_csv( "datasets/all playlists.csv", index=False, encoding='utf-8')