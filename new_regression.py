#!/usr/bin/python3

import numpy as np
import pandas as pd
from movie_feature import get_movie_feature

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

def data_normalization(data, did_rate):
    did_rate_row_sum = np.sum(did_rate,axis=1)
    did_rate_row_sum = np.array([did_rate_row_sum]).T
    data_row_sum = np.sum(data, axis=1)
    data_row_sum = np.array([data_row_sum]).T
    did_rate_row_sum = (data_row_sum == 0) * 1 + did_rate_row_sum
    mean =  data_row_sum / did_rate_row_sum
    norm = data - mean
    norm = norm * did_rate
    return norm, mean

def regression(norm,movie_feature,user_preference,did_rate):
    difference = (movie_feature.dot(user_preference.T)) * did_rate - norm
    new_movie_features = movie_feature * co - difference.dot(user_preference) * learning_rate
    new_user_preference = user_preference * co - difference.T.dot(movie_feature) * learning_rate

    return new_movie_features, new_user_preference


num_features = 19
reg_param = 1
learning_rate = 0.001
co = (1 - learning_rate * reg_param)

num_users = 943
num_movies = 1682

training_loss_mean = 0
test_loss_mean = 0
file = open("./model/testfile.txt","w") 

training_file = ['./ml-100k/u1.base', './ml-100k/u2.base', './ml-100k/u3.base', './ml-100k/u4.base', './ml-100k/u5.base']
test_file = ['./ml-100k/u1.test','./ml-100k/u2.test','./ml-100k/u3.test','./ml-100k/u4.test','./ml-100k/u5.test']
output_list = list()
for traning_f, test_f in zip(training_file, test_file):
    training_matrix = get_matrix(traning_f)
    training_rate_map = (training_matrix != 0) * 1

    test_matrix = get_matrix(test_f)
    test_rate_map = (test_matrix != 0) * 1

    norm, mean = data_normalization(training_matrix, training_rate_map)

    movie_feature = np.array(get_movie_feature(), dtype=np.float64) * 0.5
    user_preference = 0.5 + np.random.randn(num_users, num_features) * 0.01



    final_train_loss = 0
    final_test_loss = 0

    final_movie_feature = np.zeros((num_movies, num_features))
    final_user_preference = np.zeros((num_users, num_features))

    last_result = 0
    for i in range(2000):
        movie_feature, user_preference = regression(norm,movie_feature,user_preference,training_rate_map)

        # training loss
        result = movie_feature.dot(user_preference.T)
        traning_loss = result - norm
        traning_loss *=  training_rate_map
        traning_loss = np.sqrt(np.sum(traning_loss ** 2) / np.sum(training_rate_map))

        # test loss
        result += mean
        test_loss = result - test_matrix
        test_loss *= test_rate_map
        test_loss = np.sqrt(np.sum(test_loss ** 2) / np.sum(test_rate_map))

        output = "round: "+str(i)+ "  "+ str(test_loss) + "\n"
        file.write(output) 

        output_list.append(output)
        
        print("round: "+str(i)+ "  "+str(test_loss))
        if(final_test_loss < test_loss and test_loss < 1.0):
            break

        final_train_loss = traning_loss
        final_test_loss = test_loss

    training_loss_mean += final_train_loss
    test_loss_mean += final_test_loss
    final_user_preference = np.add(final_user_preference, user_preference)
    final_movie_feature = np.add(final_movie_feature, movie_feature)


print("final traning loss: "+str(training_loss_mean/5) + "   final test loss: "+str(test_loss_mean/5))


file.close() 

np.savetxt('./model/final_movie_feature.csv', final_movie_feature, delimiter=",")
np.savetxt('./model/final_user_preference.csv', final_user_preference, delimiter=",")
