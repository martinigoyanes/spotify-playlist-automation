import spotify
import pandas as pd
client_id = 'c6c3f6355e3349ce8160f0f2504e442b'
client_secret = '2da4af43872a462ab652f579aa4b9d04'
# Get spotify credentials through OAuth 2.0
spoti = spotify.Spotify(client_id, client_secret)
spoti.get_auth_code()
spoti.get_tokens()

playlist_json = spoti.get_playlist_json(id='37i9dQZF1DWXCGnD7W6WDX',
                                        fields='items(track(id,album(name),artists(name),explicit,name,duration_ms,popularity))',
                                        limit=20,
                                        offset=0)['items']
ids =[]
audio_features_json = []
for item in playlist_json:
    ids.append(item['track']['id'])

for x in ids:
    audio_features_json.append(spoti.get_audio_features(x))


playlist_df = pd.DataFrame(playlist_json)
playlist_df.to_csv('playlist.csv')

audio_features_df = pd.DataFrame(audio_features_json)
audio_features_df.to_csv('audio_features.csv')


print('hello')
