
import pandas as pd

import nltk
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv(r" spotify_millsongdata.csv")

data = data.sample(2000).drop('link',axis=1).reset_index(drop=True)

#CLEANING
data['text'].str.lower().replace(r'^\w\s', '').replace(r'\n', '', regex = True)

#TOKENIZATION:

st = PorterStemmer()

def token(txt):
    token = nltk.word_tokenize(txt)
    j = [st.stem(i) for i in token]
    return ' '.join(j)


data['text'].apply(lambda x: token(x))

vec = TfidfVectorizer(analyzer = 'word', stop_words = 'english')

matrix = vec.fit_transform(data['text'])
matrix_sim = cosine_similarity(matrix)
 

print('''MENU: 
1. Song Recommendation
2. Song Lyrics 
3. Top Ten Songs of a genre
4. Search a song by a word in lyrics
''')

option = int(input('Enter the option: '))

if option ==1:
    
    def recommendation(song_to_reco):
        idx = data[data['song']==song_to_reco].index[0]
        distance = sorted(list(enumerate(matrix_sim[idx])), reverse = True, key = lambda x: x[1])
        song_l = []
        for ids in distance[1:6]:
            song_l.append(data.iloc[ids[0]].song)
        return song_l

    print('Welcome to the Music Recommendation System')
    print('\n')
    print("while entering the song name, make sure that each word's first letter is capital")
    print('\n')
    print(data.head())
    song_input = input('Enter the song name: ')
    print(recommendation(song_input))

elif option ==2:
    print('Music Lyrics')
    print('\n')
    song_input = input('Enter the song name: ')
    idx = data[data['song']==song_input].index[0]
    print(data.iloc[idx].text)

elif option ==3:
    print("Top 10 Songs of that genre: ")
    
else:
    print('Type the correct option')

