
# This is a model for neural network with embedding leayer
# The details has been described in the report
# implementation section ----> neural network with embedding leayer 

from torch import nn
import torch
class Model(nn.Module):
    def __init__(self, num_movie, num_user, num_feature):
        super().__init__()
        self.movie_feature = nn.Embedding(num_movie, num_feature, sparse=True, max_norm=0.5)
        self.user_preference = nn.Embedding(num_user, num_feature, sparse=True, max_norm=0.5)
        self.movie_bias = nn.Embedding(num_movie, 1, max_norm=0.25)
        self.user_bias = nn.Embedding(num_user, 1, max_norm=0.25)
        self.h1_movie = nn.Linear(num_feature + 1, int(num_feature / 3))
        self.h1_user = nn.Linear(num_feature + 1, int(num_feature / 3))
        self.h1_movie.bias.norm(0.25)
        self.h1_user.bias.norm(0.25)
        
    def forward(self, movie, user):
        m = torch.cat([self.movie_feature(movie),self.movie_bias(movie)], 1)
        u = torch.cat([self.user_preference(user),self.user_bias(user)], 1)
        m = self.h1_movie(m)
        u = self.h1_user(u)
        m = torch.nn.functional.tanh(m)
        u = torch.nn.functional.tanh(u)
        x = (m*u).sum(1)
        return x