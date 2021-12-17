from search_engine import search_engine
from flask import Flask,render_template,request,redirect,url_for,session
app = Flask(__name__,template_folder='')
import requests

import pandas as pd
import random
import authorization
import numpy as np
from numpy.linalg import norm

################################################################################
#Plot
import plotly
import plotly.express as px
import json
#Function 1 Location
import numpy as np
import pandas as pd 
import sklearn
from sklearn.cluster import KMeans
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


@app.route("/", methods=['GET','POST'])
def track_id():
    output1 = None
    output2 = None
    track_name_result = ""
    if request.method == 'POST':
        track_name=request.form['track_name']
        track_id = 0
        genre1 = request.form['genre1']
        genre2 = request.form['genre2']
        genre3 = request.form['genre3']
        lyric = request.form['lyrics']

        #Load data
        df = pd.read_csv("./valence_arousal_dataset.csv")
        
        # #Create mood vector
        df["mood_vec"] = df[["valence", "energy"]].values.tolist()

        sp = authorization.authorize()
        track_name_result = search_engine(track_name)
        if len(track_name_result) not in (0,1):
             track_id = df[df['track_name'] == track_name_result]['id'].values[0]

        # In order to compute distances between two tracks, we need to transform the seperate valence and energy columns to a mood-vector column. 
        # This can be done by using df.apply() alongside a lambda function

        #The algorithm that finds similar tracks to a given input track

        #1.Crawl the track's valence and energy values from the Spotify API.
        #2.Compute the distances of the input track to each track in the reference dataset.
        #3.Sort the reference track from lowest to highest distance.
        #4.Return the n most similar tracks

        def recommend(track_id, ref_df, sp, n_recs=5):
            # Crawl valence and arousal of given track from spotify api
            track_features = sp.track_audio_features(track_id)
            track_moodvec = np.array([track_features.valence, track_features.energy])
            # print(f"mood_vec for {track_id}: {track_moodvec}")
            # Compute distances to all reference tracks
            ref_df["distances"] = ref_df["mood_vec"].apply(lambda x: norm(track_moodvec-np.array(x)))
            # Sort distances from lowest to highest
            ref_df_sorted = ref_df.sort_values(by = "distances", ascending = True)
            # If the input track is in the reference set, it will have a distance of 0, but should not be recommended
            ref_df_sorted = ref_df_sorted[ref_df_sorted["id"] != track_id]
            # Return n recommendations
            return ref_df_sorted.iloc[:n_recs]
        if track_id in df['id'].unique():
            rec = recommend(track_id=track_id, ref_df= df, sp= sp, n_recs=20)
        else:
            rec = pd.DataFrame(columns=['id', 'genre', 'track_name','artist_name','valence','energy','lyric','mood'])
        # print(rec)
        list_music = rec.copy()
        if genre1 in df['genre'].unique() or genre2 in df['genre'].unique() or genre3 in df['genre'].unique():
            list_music = list_music[list_music['genre'].isin([genre1, genre2, genre3])]
        if lyric == 'Y' or lyric == 'N':
            list_music = list_music[list_music['lyric'].isin([lyric])]
        output1 = makeObject(list_music)
    recom_list = pd.read_csv('./10000.txt', sep='\t')
    grouped_list = recom_list.groupby('song_ids').sum().reset_index()
    grouped_list.sort_values(by='listentime', ascending=False, inplace=True)
    grouped_list = grouped_list.iloc[0:10]
    linked_list = pd.read_csv('./song_data.csv')
    songs = []
    artists = []
    for i in range(len(grouped_list)):
        idx = linked_list[linked_list['song_id'] == grouped_list.iloc[i]['song_ids']]['title'].index[0]
        song = linked_list.iloc[idx]['title']
        artist = linked_list.iloc[idx]['artist_name']
        songs.append(song)
        artists.append(artist)
    grouped_list['song'] = songs
    grouped_list['artist'] = artists
    grouped_list.drop(['song_ids'], axis=1, inplace=True)
    output2 = []
    rank = 1
    for index, row in grouped_list.iterrows():
        item = {
            'artist_name': row['artist'],
            'track_name': row['song'],
            'listentime': row['listentime'],
            'rank': rank
        }
        rank += 1
        output2.append(item)
    print(output2)
    return render_template('abc.html', output1=output1, track_name_result=track_name_result, recom=output2)


def recommend():
    data = pd.read_csv('./10000.txt')



def makeObject(df):
    objectList = []
    for index, row in df.iterrows():
        item = {
            'id': row['id'],
            'artist_name': row['artist_name'],
            'track_name': row['track_name'],
            'genre': row['genre'],
            'lyric': row['lyric']
        }
        objectList.append(item)
    return objectList


if __name__ == "__main__":
    app.run(debug=True)
