from django.db import models

# Create your models here.


class User(models.Model):
    uid = models.CharField(max_length=50, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    refresh_token = models.CharField(max_length=150)
    access_token = models.CharField(max_length=150)
    expires_in = models.DateTimeField()
    token_type = models.CharField(max_length=50)
    curr_session = models.CharField(max_length=50, unique=True, null=True)
    pic_url = models.CharField(max_length=100, unique=True)
    pred_count = models.IntegerField(default=0)


class PlaylistModel(models.Model):
    uid = models.CharField(max_length=50)
    spotify_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=100,)


class AIConfig(models.Model):
    uid = models.CharField(max_length=50, unique=True)
    maxlen = models.IntegerField()
    vocab_size = models.IntegerField()


class AIEncoding(models.Model):
    uid = models.CharField(max_length=50)
    spotify_id = models.CharField(max_length=50, unique=True)
    playlist_name = models.CharField(max_length=50)
    encoding = models.IntegerField()
