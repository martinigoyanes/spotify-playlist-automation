from django.urls import path
from .views import index

app_name = 'frontend'

urlpatterns = [
    path('', index, name=''),
    path('user-page', index, name='user-page'),
    path('login', index, name='login')
]
