from rest_framework import serializers
from spotify.models import User, SongModel

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('uid', 'created_at', 'refresh_token', 'access_token', 'expires_in', 'token_type', 'model')


class SongModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = SongModel
        fields = ('name', 'artist', 'fullname', 'album', 'len_ms', 'songid',
                     'explicit', 'popularity', 'key', 'mode', 'time_signature', 'acousticness',
                         'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 
                         'speechiness', 'valence', 'tempo', 'playlist_name', 'playlist_id', 'uid')