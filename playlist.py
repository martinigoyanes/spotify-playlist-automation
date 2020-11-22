class Playlist:
    artists = []
    metadata = dict()

    def __init__(self, ):
        self.id = name
        self.video_url = video_url
        self.artists = artists
        self.downloaded = False
        self.fullname = name
        for artist in artists:
            self.fullname += f' - {artist}'
        self.outtemplate = f'{config.OUT_FOLDER}{self.fullname}.%(ext)s'
        self.fullpath = f'{config.OUT_FOLDER}{self.fullname}.mp3'
        self.thread = None
        self.failed = False
        self.metadata = {
            'cover_url': cover,
            'album': album
        }
        self.len_ms = len_ms
