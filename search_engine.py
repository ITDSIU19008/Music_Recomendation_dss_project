from difflib import SequenceMatcher
import string
import pandas as pd

track_name = pd.read_csv("./valence_arousal_dataset.csv")['track_name'].dropna().drop_duplicates()
PUNCT = string.punctuation

def similar(a,b):
    return SequenceMatcher(None, a, b).ratio()
def search_engine(name_search):
    search = {"name":name_search, "Name_Predict":[], "Ratio":[],'Priority':[]}
    for i in track_name:
        if i == name_search:
            return i
        else:
            name_search = name_search.translate(str.maketrans("", "", PUNCT))
            name_search = name_search.strip()
            # words = name_search.split(" ")  
            if len(name_search.strip()) in (0,1):
                return ""
            else:
                search['Name_Predict'].append(i)   
                search['Ratio'].append(similar(name_search,i))
                search['Priority'].append(0)
                
    search = pd.DataFrame(search)
    search = search.sort_values(by=['Priority','Ratio'],ascending=False)['Name_Predict'][0]            
    return search