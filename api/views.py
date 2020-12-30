from django.shortcuts import render
from rest_framework import viewsets
from .serializers import UserSerializer, SongModelSerializer
from spotify.models import User, SongModel

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class SongModelViewSet(viewsets.ModelViewSet):
    queryset = SongModel.objects.all()
    serializer_class = SongModelSerializer