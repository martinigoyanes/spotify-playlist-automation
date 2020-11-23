import song as Song
class Playlist:
    song_list = []
    name = ''
    def __init__(self, id, name, fields):
        self.id = id
        self.name = name
        self.fields = fields

    def json_to_playlist(self, playlist_json):
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
        for song in self.song_list:
            song.add_audio_features()
    
