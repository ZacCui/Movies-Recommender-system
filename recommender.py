import numpy as np
import pandas as pd
import sys

# data pre-processing
def get_matrix(file_name):
    data = pd.read_table(file_name,
                         delimiter='\t', header=None,
                         names=['UserID', 'MioveID', 'Rating', 'Timestamp'],
                         engine='python', encoding='latin-1')

    UserID = data.UserID
    MovieID = data.MioveID
    Rating = data.Rating
    matrix = pd.DataFrame({'user': UserID, 'movie': MovieID, 'rating': Rating})
    matrix = matrix.pivot(index='movie', columns='user', values='rating').fillna(0)
    
    temp = pd.DataFrame(data=np.zeros((num_movies+1,num_users+1)))
    temp.drop(0, inplace=True)
    temp.drop(columns=0, inplace=True)
    temp.update(matrix)
    np_matrix = temp.values
    return np_matrix

# print the name of recommend movies
def recommender(movie_data,result, user):
	listofMovies = result[user].tolist()
	listofMovies = [(i,listofMovies[i]) for i in range(len(listofMovies))]
	listofMovies.sort(key=lambda tup: tup[1], reverse=True)
	for i in range(10):
		print(i+1," Movie name:", movie_data[listofMovies[i][0]][1])


num_users = 943
num_movies = 1682

if len(sys.argv) != 2:
	print("Usage: python3 recommender.py UserId")
	exit(1)

userId = int(sys.argv[1])

if userId > num_users or userId < 1:
	print("Invalid UserId")
	exit(1)


movie_score = pd.read_csv('./model/final_movie_feature.csv',delimiter=',',engine='python', encoding='latin-1');
user_preference = pd.read_csv('./model/final_user_preference.csv',delimiter=',',engine='python', encoding='latin-1');

m_index = movie_score.columns.values
u_index = user_preference.columns.values

data = pd.read_table('./ml-100k/u.item', delimiter='\|+', header=None, engine='python', encoding='latin-1')
movie_data = data.values
movie_data = movie_data[:,:4]
movie_score = movie_score.values
user_preference = user_preference.values

m_last = movie_score[-1]
u_last = user_preference[-1]

movie_score = np.insert(movie_score,0, m_index, 0)
user_preference = np.insert(user_preference, 0, u_index, 0)

files = ['./ml-100k/u1.base', './ml-100k/u2.base', './ml-100k/u3.base', './ml-100k/u4.base', 
'./ml-100k/u5.base','./ml-100k/u1.test','./ml-100k/u2.test','./ml-100k/u3.test','./ml-100k/u4.test','./ml-100k/u5.test']
flag = False
for file in files:
    training_matrix = get_matrix(file)
    training_rate_map = (training_matrix != 0) * 1
    if(not flag):
    	not_did_rate = training_rate_map
    else:
    	not_did_rate = np.add(not_did_rate,training_rate_map)
    	
not_did_rate = (not_did_rate == 0) * 1

# get rating martix and filter the rated movie for each user
result = user_preference.dot(movie_score.T) * not_did_rate.T

recommender(movie_data, result, userId - 1)


