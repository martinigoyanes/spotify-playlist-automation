import datetime
import multiprocessing
import pandas as pd
import time
from .spotify import get_audio_features, put_song_in_playlist, get_user_tracks_json
import backend.util as util
import backend.AIModel as AIModel
import backend.models as models
class UserTracker(multiprocessing.Process):
    def __init__(self, uid):
        multiprocessing.Process.__init__(self)
        self.user = util.get_user(uid=uid)
        self.last_added = None
    
    def get_last_date(self):
        # Get user date of last liked 
        tracks_json = get_user_tracks_json(self.user, limit=20)
        item = tracks_json['items'][0]
        last_added = item['added_at']
        last_added = datetime.datetime.strptime(last_added, "%Y-%m-%dT%H:%M:%SZ")
        self.last_added = last_added
    
    def run(self):
        self.start_tracking()

    def start_tracking(self):
        self.get_last_date()

        while True:
            tracks_json = get_user_tracks_json(self.user, limit=20)
            for item in tracks_json['items']:
                added_at = item['added_at']
                added_at = datetime.datetime.strptime(added_at, "%Y-%m-%dT%H:%M:%SZ")
                if added_at > self.last_added:
                    audio_features_json = get_audio_features(item['track']['id'], self.user)
                    song = {
                        'artist': item['track']['artists'][0]['name'],
                        'len_ms': item['track']['duration_ms'],
                        'explicit': item['track']['explicit'],
                        'popularity': item['track']['popularity'],
                        'key': audio_features_json['key'],
                        'mode': audio_features_json['mode'],
                        'time_signature': audio_features_json['time_signature'],
                        'acousticness': audio_features_json['acousticness'],
                        'danceability': audio_features_json['danceability'],
                        'energy': audio_features_json['energy'],
                        'instrumentalness': audio_features_json['instrumentalness'],
                        'liveness': audio_features_json['liveness'],
                        'loudness': audio_features_json['loudness'],
                        'speechiness': audio_features_json['speechiness'],
                        'valence': audio_features_json['valence'],
                        'tempo': audio_features_json['tempo']
                    }
                    df = pd.DataFrame(song, index=[0])
                    model = AIModel.AIModel(self.user.uid)
                    model.load_user_model()
                    playlist_id = model.predict(df)
                    song_name = item['track']['name']
                    song_uri = item['track']['uri']

                    playlist = models.PlaylistModel.objects.filter(spotify_id=playlist_id)[0]
                    print(f'Classified {song_name} into {playlist.name}')
                    resp = put_song_in_playlist(self.user, playlist_id, [song_uri])
            self.get_last_date()
            time.sleep(10) #seconds
