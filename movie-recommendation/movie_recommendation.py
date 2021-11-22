from datetime import time
import numpy as np
from numpy.lib.function_base import vectorize
import pandas as pd

import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the data from csv file to a pandas dataframe
movies_data = pd.read_csv("./data/movies.csv")
# print(movies_data.head())
# print(movies_data.shape)

#  Selecting the relavant features for recommendation

selected_features = ['genres', 'keywords' , 'tagline', 'cast', 'director']
# print(selected_features)

# replacing the null values with null string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    
# Combining all the 5 selected features
combined_features = movies_data['genres'] +' '+ movies_data['keywords']+ ' '+movies_data['tagline']+ ' '+ movies_data['cast']+ ' '+ movies_data['director']
# print(combined_features)

# converting the text data to feature vectors

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
# print(feature_vectors)

# ----------------- cosine similarity -----------------------

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vectors)
# print(similarity)

# Getting the movie name from the user

movie_name = input('Enter your favourite movie name: ')

# Creating a list with all the movie names given in the dataset
list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)

# Finding the closr match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)

close_match = find_close_match[0]

# finding the index of the mocie with title
index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

# getting a list of similar movies
similarity_score = list(enumerate(similarity[index_of_the_movie]))
# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score , key= lambda x:x[1] , reverse = True)

# print the name of similar movies based pn the index
print('Movied suggested for you : \n')

i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if (i < 30):
        print(i , '.' ,title_from_index)
        i+=1