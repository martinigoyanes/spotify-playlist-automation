import webbrowser
import requests
import base64
import time

from urllib.parse import urlencode, urlsplit


def get_auth_code(self):
    auth_endpoint = 'https://accounts.spotify.com/authorize'
    auth_body = {
        'client_id': self.id,
        'response_type': 'code',
        'redirect_uri': 'https://www.spotify.com/es/',
        'scope': 'user-library-read'
    }
    auth_body_urlencoded = urlencode(auth_body)
    auth_url = f"{auth_endpoint}?{auth_body_urlencoded}"
    # webbrowser.open(auth_url)
    # self.auth_code = input('Authentication code:\n')
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get(auth_url)
    code_url = ''
    code = urlsplit(code_url)
    while 'code=' not in code.query:
        code_url = driver.current_url
        code = urlsplit(code_url)
    self.auth_code = code.query.replace('code=', '')
    driver.quit()


def get_tokens(self):
    tokens_endpoint = 'https://accounts.spotify.com/api/token'
    tokens_body = {
        'grant_type': 'authorization_code',
        'code': self.auth_code,
        'redirect_uri': 'https://www.spotify.com/es/'
    }
    client_creds_b64 = base64.b64encode(
        f'{self.id}:{self.secret}'.encode())
    tokens_headers = {
        # Authorization: Basic *<base64 encoded client_id:client_secret>*
        'Authorization': f'Basic {client_creds_b64.decode()}'
    }
    resp = requests.post(
        tokens_endpoint, data=tokens_body, headers=tokens_headers)
    tokens_resp_data = resp.json()
    self.access_token = tokens_resp_data['access_token']
    self.refresh_token = tokens_resp_data['refresh_token']
    self.access_token_expires = tokens_resp_data['expires_in']


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


