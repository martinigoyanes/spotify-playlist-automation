class Playlist:
    def __init__(self, id, name, fields):
        self.id = id
        self.name = name
        self.fields = fields
        self.song_list = []

    def json_to_playlist(self, playlist_json):
        import song as Song
        usefull_songs = (
            item for item in playlist_json['items'] if item['track']['id'] is not None)
        for item in usefull_songs:
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
                             len_ms=song_len_ms, explicit=song_explicit,
                             popularity=song_popularity, id=song_id)
            self.song_list.append(song)

    def add_audio_features(self):
        import config
        import math
        from decimal import Decimal, getcontext
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
        if n_requests > 1:
            for n in range(1, n_requests_low + 1):
                audio_features_json = config.spoti.get_audio_features_several(
                    ids[((n-1)*100):(n*100)])
                for audio_features, song in zip(audio_features_json['audio_features'], self.song_list[((n-1)*100):(n*100)]):
                    song.add_audio_features(audio_features)
                # Partial request of < 100 (last request)
                if n == (n_requests_low + 1):
                    audio_features_json = config.spoti.get_audio_features_several(
                        ids[(n*100):((n*100)+partial_request_len)])
                    for audio_features, song in zip(audio_features_json['audio_features'], self.song_list):
                        song.add_audio_features(audio_features)
        else:
            # Partial request of < 100
            audio_features_json = config.spoti.get_audio_features_several(
                ids[0:partial_request_len])
            for audio_features, song in zip(audio_features_json['audio_features'], self.song_list):
                song.add_audio_features(audio_features)

    def to_csv(self):
        import csv
        import os

        if not os.path.exists('../datasets'):
            os.makedirs('../datasets')

        filename = f'../datasets/{self.name}-{self.id}.csv'
        try:
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Artists', 'Album', 'Duration(ms)', 'Explicit', 'Popularity', 'Key', 'Mode', 'Time Signature', 'Acousticness',
                                 'Danceability', 'Energy', 'Instrumentalness', 'Liveness', 'Loudness', 'Speechiness', 'Valence', 'Tempo', 'Playlist'])
                for song in self.song_list:
                    writer.writerow([song.name, ",".join(song.artists), song.album, song.len_ms, song.explicit, song.popularity,
                                     song.key, song.mode, song.time_signature, song.acousticness, song.danceability,
                                     song.energy, song.instrumentalness, song.liveness, song.loudness, song.speechiness,
                                     song.valence, song.tempo, self.name])
        except BaseException as e:
            print('BaseException:', filename)
