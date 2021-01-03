class Song:
    def __init__(self, name, artist, album, len_ms, id, explicit, popularity):
        self.name = name
        self.artist = artist
        self.album = album
        self.len_ms = len_ms
        self.id = id
        self.explicit = explicit
        self.popularity = popularity
        self.key = None
        self.mode = None
        self.time_signature = None
        self.acousticness = None
        self.danceability = None
        self.energy = None
        self.instrumentalness = None
        self.liveness = None
        self.loudness = None
        self.speechiness = None
        self.valence = None
        self.tempo = None

    def add_audio_features(self, audio_features):
        if audio_features is not None:
            self.key = audio_features['key']
            self.mode = audio_features['mode']
            self.time_signature = audio_features['time_signature']
            self.acousticness = audio_features['acousticness']
            self.danceability = audio_features['danceability']
            self.energy = audio_features['energy']
            self.instrumentalness = audio_features['instrumentalness']
            self.liveness = audio_features['liveness']
            self.loudness = audio_features['loudness']
            self.speechiness = audio_features['speechiness']
            self.valence = audio_features['valence']
            self.tempo = audio_features['tempo']
