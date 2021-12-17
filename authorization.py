import tekore as tk

def authorize():
    CLIENT_ID = "05bbd92fa9594ef687995e872b11d7a5"
    CLIENT_SECRET = "3adf77dd975843dc81a3e545828803ce"
    app_token = tk.request_client_token(CLIENT_ID, CLIENT_SECRET)
    return tk.Spotify(app_token)