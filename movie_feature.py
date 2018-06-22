import numpy as np
import pandas as pd

def get_movie_feature():
    data = pd.read_table('./ml-100k/u.item', delimiter='\|+', header=None, engine='python', encoding='latin-1')

    matrix = data.values
    matrix = matrix[:,4:]
    matrix = (matrix == 1) *1
    return matrix
    