import requests
import base64
import time

from urllib.parse import urlencode, urlsplit

def get_user_tracks_json(user, limit=1, offset=None):
    user_tracks_endpoint = 'https://api.spotify.com/v1/me/tracks'
    user_tracks_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }
    user_tracks_body = {
        'limit': limit
    }
    if offset is not None:
        user_tracks_body.update({'offset': offset})

    user_tracks_body_urlencoded = urlencode(user_tracks_body)
    user_tracks_url = f"{user_tracks_endpoint}?{user_tracks_body_urlencoded}"
    resp = requests.get(user_tracks_url, headers=user_tracks_headers)

    return resp.json()


def get_user_playlists(user, limit=50, offset=None):
    user_playlists_endpoint = 'https://api.spotify.com/v1/me/playlists'
    user_playlists_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }
    user_playlists_body = {
        'limit': limit
    }
    if offset is not None:
        user_playlists_body.update({'offset': offset})

    user_playlists_body_urlencoded = urlencode(user_playlists_body)
    user_playlists_url = f"{user_playlists_endpoint}?{user_playlists_body_urlencoded}"
    resp = requests.get(user_playlists_url, headers=user_playlists_headers)

    return resp.json()


def get_playlist_json(user, id, fields=None, limit=1, offset=None):
    playlist_endpoint = f'https://api.spotify.com/v1/playlists/{id}/tracks'
    playlist_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }
    playlist_body = {
        'limit': limit,
    }

    if offset is not None:
        playlist_body.update({'offset': offset})
    if fields is not None:
        playlist_body.update({'fields': fields})

    playlist_body_urlencoded = urlencode(playlist_body)
    playlist_url = f"{playlist_endpoint}?{playlist_body_urlencoded}"
    resp = requests.get(playlist_url, headers=playlist_headers)

    return resp.json()


def put_song_in_playlist(user, playlist_id, uris):
    playlist_endpoint = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    playlist_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }
    playlist_body = {
        'uris': ",".join(uris),
    }

    playlist_body_urlencoded = urlencode(playlist_body)
    playlist_url = f"{playlist_endpoint}?{playlist_body_urlencoded}"
    resp = requests.post(playlist_url, headers=playlist_headers)

    return resp.json()


def get_audio_features_several(ids, user):
    audio_features_endpoint = f'https://api.spotify.com/v1/audio-features'
    audio_features_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }
    # Generate comma separated list
    audio_features_body = {
        'ids': ",".join(ids),
    }

    audio_features_body_urlencoded = urlencode(audio_features_body)
    audio_features_url = f"{audio_features_endpoint}?{audio_features_body_urlencoded}"
    resp = requests.get(audio_features_url, headers=audio_features_headers)

    return resp.json()


def get_audio_features(id, user):
    audio_features_endpoint = f'https://api.spotify.com/v1/audio-features/{id}'
    audio_features_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {user.access_token}'
    }

    resp = requests.get(audio_features_endpoint,
                        headers=audio_features_headers)

    return resp.json()
    
def get_user_profile_pic(access_token):
    user_profile_endpoint = f'https://api.spotify.com/v1/me'
    user_profile_headers = {
        # Authorization: Bearer {your access token}
        'Authorization': f'Bearer {access_token}'
    }

    resp = requests.get(user_profile_endpoint,
                        headers=user_profile_headers)

    return resp.json()


def pull_playlist(user, playlist):
    limit = 100
    offset = 0
    while True:
        playlist_json = get_playlist_json(id=playlist.id,
                                          user=user,
                                          fields=playlist.fields,
                                          limit=100,
                                          offset=offset)
        playlist.json_to_playlist(playlist_json)
        offset = offset + limit
        # Last couple of songs, we exit the loop, there are not more songs in the playlist
        if len(playlist_json['items']) < limit:
            break