import spotify
client_id = None
client_secret = None
spoti = None
standard_fields = None

def init_globals():
    global spoti, standard_fields
    client_id = 'c6c3f6355e3349ce8160f0f2504e442b'
    client_secret = '2da4af43872a462ab652f579aa4b9d04'
    standard_fields = 'items(track(id, album(name), artists(name), explicit, name, duration_ms, popularity))'

    # Get spotify credentials through OAuth 2.0
    spoti = spotify.Spotify(client_id, client_secret)
    spoti.get_auth_code()
    spoti.get_tokens()
