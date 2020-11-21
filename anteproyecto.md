# Static model
## Cons
dataset = Artista, Nombre cancion, duracion cancion, album, label(playlist)
## Pros
Tenemos 3000 playlist ya en .csv o .json

# Dynamic model
## Cons
Crear nuestro propio dataset a partir del spotify api -> **data gathering**

## Pros
dataset = \
&nbsp; &nbsp; Artista, Nombre cancion, duracion cancion, album, acousticness,\
&nbsp; &nbsp; danceability, energy, instrumentalness, liveness, loudness,  
&nbsp; &nbsp; speechiness, valence,
tempo, (mode, time-signature,.......... ),\
&nbsp; &nbsp; **playlist**

# 0-100 Knowledge model
Modelo que coge los datos del usuario y se entrena UNICAMENTE sobre ellos y a partir de ahi ya empieza a funcionar para ese usuario en concreto
## Pros
Un unico modelo entrenado con datos del usuario y ya
## Cons 
Si el usuario tiene apenas playlist creadas, poca diversidad en sus playlists, ... -> **modelo underfitted**


# 70-100 Knowledge model
Modelo que se entrena  una vez con BASTANTES datos hasta que es eficiente. Posteriormente, cada usuario usa este modelo y lo REENTRENA con sus datos personales para su uso particular
## Pros
No es problema que el usuario tenga pocas playlist o poca variedad, modelo robusto siempre ya que tiene conocimiento guardado
## Cons
Crear modelo base con suficientes datos para que sea robusto y luego reentrenar para cada usuario
    
# Conclusiones
Martin -> {Dynamic Model, 70-100 Knowledge Model}
Jaime  -> {xxx,yyy}