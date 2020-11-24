class Playlist:
    def __init__(self, id, name, fields):
        self.id = id
        self.name = name
        self.fields = fields
        self.song_list = []

    def json_to_playlist(self, playlist_json):
        import song as Song
        for item in playlist_json['items']:
            song_name = item['track']['name']
            song_artists = []
            for artist in item['track']['artists']:
                song_artists.append(artist['name'])
            song_album = item['track']['album']['name']
            song_len_ms = item['track']['duration_ms']
            song_id = item['track']['id']
            song_explicit = item['track']['explicit']
            song_popularity = item['track']['popularity']
            song = Song.Song(name=song_name, artists=song_artists, album=song_album, 
                                len_ms=song_len_ms, explicit = song_explicit, 
                                    popularity = song_popularity, id = song_id)
            self.song_list.append(song)
    def add_audio_features(self):
        import config
        ids = [song.id for song in self.song_list]
        audio_features_json = config.spoti.get_audio_features_several(ids)
        for audio_features, song in zip(audio_features_json['audio_features'], self.song_list):
           song.add_audio_features(audio_features)

    def to_csv(self):
        import csv
        import os
        
        if not os.path.exists('datasets'):
            os.makedirs('datasets')

        filename = f'datasets/{self.name}-{self.id}.csv'
        try:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Artists', 'Album', 'Duration(ms)', 'Explicit', 'Popularity', 'Key', 'Mode', 'Time Signature', 'Acousticness',\
                                 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Playlist'])
                for song in self.song_list:
                    writer.writerow([song.name, ",".join(song.artists), song.album, song.len_ms, song.explicit, song.popularity,
                                    song.key, song.mode, song.time_signature, song.acousticness, song.danceability,
                                    song.energy, song.instrumentalness, song.liveness, song.loudness, song.speechiness,
                                    song.valence, song.tempo, self.name])
        except BaseException as e:
            print('BaseException:', filename)