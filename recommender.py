# Import AL/ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

# Import the movies dataset
movies = pd.read_csv('ml-latest-small/movies.csv')
print(movies.head())

# Import the movie ratings dataset
ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(ratings.head())
