import django
django.setup()
from .models import User
from django.utils import timezone
from datetime import timedelta
from .credentials import CLIENT_ID, CLIENT_SECRET
from requests import post, put, get
from multiprocessing import Process
import datetime
import time
import sys
import pandas as pd

from .spotify import get_user_playlists, pull_playlist, get_user_tracks_json, get_audio_features, put_song_in_playlist
from .playlist import Playlist

import backend.AIModelTrainer as AIModelTrainer

def get_user(uid=None, session_key=None):
    user_tokens = User.objects.filter(
        uid=uid) if uid else User.objects.filter(curr_session=session_key)
    if user_tokens.exists():
        return user_tokens[0]
    else:
        return None


def update_or_create_user(curr_session, access_token, token_type, expires_in, refresh_token):

    response = get('https://api.spotify.com/v1/me', headers={
        'Authorization': f'Bearer {access_token}'
    }).json()
    uid = response.get('id')

    user = get_user(uid=uid)
    expires_in = timezone.now() + timedelta(seconds=expires_in)

    if user:
        user.access_token = access_token
        user.refresh_token = refresh_token
        user.expires_in = expires_in
        user.token_type = token_type
        user.curr_session = curr_session
        user.save(update_fields=['access_token',
                                 'refresh_token', 'expires_in', 'token_type', 'curr_session'])
    else:
        create_user(curr_session, uid, access_token,
                    refresh_token, token_type, expires_in)
        # tokens = User(uid=uid, access_token=access_token,
        #               refresh_token=refresh_token, token_type=token_type, expires_in=expires_in)
        # tokens.save()


def create_user(curr_session, uid, access_token, refresh_token, token_type, expires_in):
    user = User(uid=uid, access_token=access_token,
                refresh_token=refresh_token,
                token_type=token_type, expires_in=expires_in, curr_session=curr_session)
    user.save()

    model_trainer_process = AIModelTrainer.AIModelTrainer(user.uid) 
    model_trainer_process.start()
    
def is_spotify_authenticated(session_key):
    user = get_user(session_key=session_key)
    if user:
        expiry = user.expires_in
        if expiry <= timezone.now():
            refresh_spotify_token(user)

        return True

    return False


def refresh_spotify_token(user):
    refresh_token = user.refresh_token

    response = post('https://accounts.spotify.com/api/token', data={
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET
    }).json()

    access_token = response.get('access_token')
    token_type = response.get('token_type')
    expires_in = response.get('expires_in')
    refresh_token = response.get('refresh_token')

    update_or_create_user(
        user.curr_session, access_token, token_type, expires_in, refresh_token)
